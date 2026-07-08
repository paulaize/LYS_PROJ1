"""Build a manual IHC threshold review dashboard.

This is intentionally a review aid, not an approval engine. It summarizes the
threshold sweep CSV, filters to plausible candidate thresholds, and creates a
static HTML page where Paul can inspect section QC thumbnails and threshold
overlays before signing off a threshold.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, load_config  # noqa: E402


@dataclass(frozen=True)
class SectionRef:
    source_animal: str
    source_role: str
    panel: str
    section_id: str
    selected: bool
    excluded: bool
    notes: str


@dataclass(frozen=True)
class ThresholdSummary:
    threshold: float
    target_mean: float
    target_min: float
    target_max: float
    control_mean: float
    control_min: float
    control_max: float
    separation_fold: float
    candidate: bool
    reason: str


def _selection_sets(panel_cfg: dict) -> tuple[set[str], set[str]]:
    selection = panel_cfg.get("section_selection") or {}
    selected = {str(v) for v in selection.get("selected_section_ids") or []}
    excluded = {str(v) for v in selection.get("excluded_section_ids") or []}
    return selected, excluded


def _configured_sections(cfg: Config, panel: str) -> list[SectionRef]:
    refs: list[SectionRef] = []

    def add_source(source_cfg: Config, source_role: str) -> None:
        panel_cfg = source_cfg.panel_config(panel)
        selected, excluded = _selection_sets(panel_cfg)
        all_sections = selected | excluded
        if not all_sections:
            qupath = source_cfg.animal.get("ihc", {}).get("qupath", {})
            all_sections = {
                str(v) for v in (qupath.get("section_ids_by_series_index") or {}).values()
            }
        notes = str((panel_cfg.get("section_selection") or {}).get("notes") or "")
        for section_id in sorted(all_sections):
            refs.append(
                SectionRef(
                    source_animal=source_cfg.animal_id,
                    source_role=source_role,
                    panel=panel,
                    section_id=section_id,
                    selected=section_id in selected,
                    excluded=section_id in excluded,
                    notes=notes,
                )
            )

    add_source(cfg, "target")
    for control in cfg.animal.get("ihc", {}).get("threshold_controls") or []:
        control_config = control.get("config")
        if not control_config:
            continue
        control_cfg = load_config(control_config, repo_root=cfg.repo_root)
        panel_cfg = control_cfg.panel_config(panel)
        if panel_cfg.get("exclude_from_threshold_calibration"):
            continue
        add_source(control_cfg, str(control.get("role", "control")))

    return refs


def _read_sweep_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing threshold sweep CSV: {path}. Run ihc-threshold-sweeps first."
        )
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"Threshold sweep CSV is empty: {path}")
    required = {"source_role", "threshold", "igg_fitc_pct_positive_area"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Threshold sweep CSV missing columns {sorted(missing)}: {path}")
    return rows


def summarize_thresholds(
    rows: list[dict[str, str]],
    *,
    max_control_pct: float = 1.0,
    min_target_pct: float = 0.05,
    min_target_control_fold: float = 5.0,
) -> list[ThresholdSummary]:
    grouped: dict[float, dict[str, list[float]]] = {}
    for row in rows:
        threshold = float(row["threshold"])
        role = row["source_role"]
        value = float(row["igg_fitc_pct_positive_area"])
        role_key = "target" if role == "target" else "control"
        grouped.setdefault(threshold, {"target": [], "control": []})[role_key].append(value)

    summaries: list[ThresholdSummary] = []
    for threshold in sorted(grouped):
        target = grouped[threshold]["target"]
        control = grouped[threshold]["control"]
        if not target or not control:
            reason = "missing target or control rows"
            summaries.append(
                ThresholdSummary(
                    threshold,
                    _mean(target),
                    _min(target),
                    _max(target),
                    _mean(control),
                    _min(control),
                    _max(control),
                    math.nan,
                    False,
                    reason,
                )
            )
            continue
        target_mean = _mean(target)
        control_mean = _mean(control)
        if control_mean == 0:
            fold = math.inf if target_mean > 0 else 0.0
        else:
            fold = target_mean / control_mean
        reasons: list[str] = []
        if control_mean > max_control_pct:
            reasons.append(f"control mean > {max_control_pct:g}%")
        if target_mean < min_target_pct:
            reasons.append(f"target mean < {min_target_pct:g}%")
        if fold < min_target_control_fold:
            reasons.append(f"target/control fold < {min_target_control_fold:g}")
        candidate = not reasons
        summaries.append(
            ThresholdSummary(
                threshold=threshold,
                target_mean=target_mean,
                target_min=min(target),
                target_max=max(target),
                control_mean=control_mean,
                control_min=min(control),
                control_max=max(control),
                separation_fold=fold,
                candidate=candidate,
                reason="candidate" if candidate else "; ".join(reasons),
            )
        )
    return summaries


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _min(values: list[float]) -> float:
    return min(values) if values else math.nan


def _max(values: list[float]) -> float:
    return max(values) if values else math.nan


def _threshold_label(threshold: float) -> str:
    if threshold == round(threshold):
        return str(int(threshold))
    return f"{threshold:.3f}".rstrip("0").rstrip(".")


def _qc_thumbnail(cfg: Config, ref: SectionRef) -> Path | None:
    path = (
        cfg.work_dir()
        / "ihc_section_qc"
        / f"panel_{ref.panel}"
        / f"{ref.source_animal}_{ref.source_role}"
        / ref.section_id
        / "section_qc_composite.png"
    )
    return path if path.exists() else None


def _review_dir(cfg: Config, ref: SectionRef) -> Path:
    return (
        cfg.work_dir()
        / "ihc_threshold_review"
        / f"panel_{ref.panel}"
        / f"{ref.source_animal}_{ref.source_role}"
        / ref.section_id
    )


def _rel_url(path: Path, html_path: Path) -> str:
    rel = Path(os.path.relpath(path, start=html_path.parent))
    return quote(rel.as_posix())


def _fmt(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def build_dashboard(
    config_path: str | Path,
    *,
    panel: str,
    max_control_pct: float = 1.0,
    min_target_pct: float = 0.05,
    min_target_control_fold: float = 5.0,
) -> tuple[Path, Path, list[ThresholdSummary]]:
    cfg = load_config(config_path)
    panel = str(panel)
    sweep_csv = cfg.work_dir() / f"ihc_threshold_sweep_panel_{panel}.csv"
    rows = _read_sweep_rows(sweep_csv)
    summaries = summarize_thresholds(
        rows,
        max_control_pct=max_control_pct,
        min_target_pct=min_target_pct,
        min_target_control_fold=min_target_control_fold,
    )
    candidates = [s for s in summaries if s.candidate]
    if not candidates:
        raise ValueError(
            "No plausible threshold candidates found. Adjust review criteria or rerun sweeps."
        )

    refs = _configured_sections(cfg, panel)
    out_dir = cfg.work_dir() / "ihc_manual_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"panel_{panel}_threshold_review.html"
    template_path = out_dir / f"panel_{panel}_threshold_signoff_template.json"
    template = _decision_template(cfg, panel, refs, candidates)
    template_path.write_text(json.dumps(template, indent=2) + "\n")
    html_path.write_text(_render_html(cfg, panel, refs, summaries, candidates, html_path, template))
    return html_path, template_path, summaries


def _decision_template(
    cfg: Config,
    panel: str,
    refs: list[SectionRef],
    candidates: list[ThresholdSummary],
) -> dict:
    return {
        "animal_id": cfg.animal_id,
        "panel": panel,
        "review_type": "ihc_igg_fitc_threshold_manual_review",
        "review_date": date.today().isoformat(),
        "reviewer": cfg.pipeline.get("ihc", {})
        .get("threshold_calibration", {})
        .get("reviewer", ""),
        "status": "manual_review_pending",
        "candidate_thresholds": [s.threshold for s in candidates],
        "selected_threshold": None,
        "approved": False,
        "section_decisions": [
            {
                "source_animal": ref.source_animal,
                "source_role": ref.source_role,
                "section_id": ref.section_id,
                "usable_for_threshold": ref.selected and not ref.excluded,
                "notes": "",
            }
            for ref in refs
        ],
        "notes": (
            "Choose threshold after inspecting candidate overlays. Mark folds/tears/debris "
            "as artifact_exclude in QuPath before final export."
        ),
    }


def _render_html(
    cfg: Config,
    panel: str,
    refs: list[SectionRef],
    summaries: list[ThresholdSummary],
    candidates: list[ThresholdSummary],
    html_path: Path,
    template: dict,
) -> str:
    candidate_values = {s.threshold for s in candidates}
    title = f"{cfg.animal_id} Panel {panel} IgG-FITC Manual Threshold Review"
    sections_html = "\n".join(_section_card(cfg, ref, html_path) for ref in refs)
    summary_rows = "\n".join(_summary_row(s) for s in summaries)
    candidate_panels = "\n".join(
        _candidate_threshold_panel(cfg, panel, refs, s.threshold, html_path)
        for s in summaries
        if s.threshold in candidate_values
    )
    candidate_radios = "\n".join(
        f"""
        <label class="radio-row">
          <input type="radio" name="threshold" value="{html.escape(_threshold_label(s.threshold))}">
          <span>{html.escape(_threshold_label(s.threshold))}</span>
          <small>target mean {_fmt(s.target_mean)}%, control mean {_fmt(s.control_mean)}%</small>
        </label>
        """
        for s in candidates
    )
    template_json = json.dumps(template).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1d1d1f; background: #f5f5f7; }}
    header {{ padding: 24px 32px; background: #ffffff; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 10; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 32px 0 12px; font-size: 18px; }}
    main {{ padding: 0 32px 40px; }}
    .notice {{ padding: 12px 14px; background: #fff8df; border: 1px solid #e7cf77; border-radius: 6px; max-width: 1120px; }}
    table {{ border-collapse: collapse; background: white; width: 100%; max-width: 1120px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e6e6e6; text-align: right; font-size: 13px; }}
    th:first-child, td:first-child, td:last-child {{ text-align: left; }}
    .candidate {{ background: #e9f7ef; }}
    .section-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 10px; }}
    .card img {{ width: 100%; height: auto; display: block; background: #111; border-radius: 4px; }}
    .card label {{ display: flex; gap: 8px; align-items: center; font-weight: 600; margin-bottom: 8px; }}
    .meta {{ color: #666; font-size: 12px; margin: 6px 0 0; }}
    .threshold-block {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 14px; margin-bottom: 18px; }}
    .thumb-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }}
    .radio-row {{ display: flex; align-items: baseline; gap: 10px; margin: 8px 0; }}
    textarea {{ width: 100%; max-width: 720px; min-height: 90px; }}
    button {{ border: 0; background: #1d4ed8; color: white; padding: 10px 14px; border-radius: 5px; cursor: pointer; font-weight: 600; }}
    code {{ background: #eee; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div>Candidate thresholds are filtered from the sweep table. This page is a review aid, not final approval.</div>
</header>
<main>
  <h2>Manual Decision</h2>
  <div class="notice">
    <p><strong>Review order:</strong> confirm decent sections, inspect only plausible thresholds, choose one threshold, then download a decision JSON.</p>
    <p>Before final export, folds/tears/debris should be annotated in QuPath as <code>artifact_exclude</code>. Current direct-image thumbnails may not include saved project annotations.</p>
  </div>
  <h2>Threshold Choice</h2>
  <div class="card" style="max-width: 720px;">
    {candidate_radios}
    <label for="review_notes"><strong>Review notes</strong></label><br>
    <textarea id="review_notes" placeholder="Notes about fold artifacts, signal pattern, rejected thresholds, and final rationale"></textarea><br><br>
    <button type="button" onclick="downloadDecision()">Download decision JSON</button>
    <p class="meta">Template data: <code id="template-name">panel_{html.escape(panel)}_threshold_signoff_template.json</code></p>
  </div>

  <h2>Threshold Summary</h2>
  <table>
    <thead>
      <tr><th>Threshold</th><th>Target mean %</th><th>Target range %</th><th>Control mean %</th><th>Control range %</th><th>Fold</th><th>Status</th></tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2>Section QC Thumbnails</h2>
  <div class="section-grid">{sections_html}</div>

  <h2>Candidate Threshold Overlays</h2>
  {candidate_panels}
</main>
<script>
const template = {template_json};
function downloadDecision() {{
  const threshold = document.querySelector('input[name="threshold"]:checked');
  const sections = Array.from(document.querySelectorAll('.section-use')).map(el => ({{
    source_animal: el.dataset.sourceAnimal,
    source_role: el.dataset.sourceRole,
    section_id: el.dataset.sectionId,
    usable_for_threshold: el.checked,
    notes: ''
  }}));
  const decision = {{
    ...template,
    status: 'manual_review_completed',
    selected_threshold: threshold ? Number(threshold.value) : null,
    approved: Boolean(threshold),
    section_decisions: sections,
    notes: document.getElementById('review_notes').value
  }};
  const blob = new Blob([JSON.stringify(decision, null, 2) + '\\n'], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'panel_{html.escape(panel)}_threshold_review_decision.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}
</script>
</body>
</html>
"""


def _summary_row(summary: ThresholdSummary) -> str:
    klass = "candidate" if summary.candidate else ""
    fold = _fmt(summary.separation_fold, 2)
    return f"""
    <tr class="{klass}">
      <td>{html.escape(_threshold_label(summary.threshold))}</td>
      <td>{_fmt(summary.target_mean)}</td>
      <td>{_fmt(summary.target_min)}-{_fmt(summary.target_max)}</td>
      <td>{_fmt(summary.control_mean)}</td>
      <td>{_fmt(summary.control_min)}-{_fmt(summary.control_max)}</td>
      <td>{fold}</td>
      <td>{html.escape(summary.reason)}</td>
    </tr>
    """


def _section_card(cfg: Config, ref: SectionRef, html_path: Path) -> str:
    thumb = _qc_thumbnail(cfg, ref)
    checked = "checked" if ref.selected and not ref.excluded else ""
    state = "selected" if ref.selected else "excluded" if ref.excluded else "unclassified"
    img = (
        f'<img src="{_rel_url(thumb, html_path)}" alt="section QC thumbnail">'
        if thumb
        else '<div class="meta">No section-QC thumbnail found</div>'
    )
    return f"""
    <div class="card">
      <label>
        <input class="section-use" type="checkbox" {checked}
          data-source-animal="{html.escape(ref.source_animal)}"
          data-source-role="{html.escape(ref.source_role)}"
          data-section-id="{html.escape(ref.section_id)}">
        <span>{html.escape(ref.source_animal)} {html.escape(ref.source_role)} {html.escape(ref.section_id)}</span>
      </label>
      {img}
      <div class="meta">Config state: {html.escape(state)}</div>
    </div>
    """


def _candidate_threshold_panel(
    cfg: Config,
    panel: str,
    refs: list[SectionRef],
    threshold: float,
    html_path: Path,
) -> str:
    label = _threshold_label(threshold)
    cards: list[str] = []
    for ref in refs:
        if not ref.selected or ref.excluded:
            continue
        review_dir = _review_dir(cfg, ref)
        raw = review_dir / "fitc_raw.png"
        overlay = review_dir / f"threshold_{label}.png"
        if not overlay.exists():
            continue
        raw_html = (
            f'<img src="{_rel_url(raw, html_path)}" alt="raw FITC">'
            if raw.exists()
            else '<div class="meta">No raw thumbnail</div>'
        )
        cards.append(
            f"""
            <div class="card">
              <strong>{html.escape(ref.source_animal)} {html.escape(ref.source_role)} {html.escape(ref.section_id)}</strong>
              <div class="meta">Raw FITC</div>
              {raw_html}
              <div class="meta">Threshold {html.escape(label)}</div>
              <img src="{_rel_url(overlay, html_path)}" alt="threshold overlay">
            </div>
            """
        )
    return f"""
    <section class="threshold-block">
      <h3>Threshold {html.escape(label)}</h3>
      <div class="thumb-grid">{''.join(cards) if cards else '<p>No thumbnails found for this threshold.</p>'}</div>
    </section>
    """


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an IHC threshold review dashboard.")
    parser.add_argument("--config", required=True, help="config/animals/<id>.yml")
    parser.add_argument("--panel", default="A", help="IHC panel to review")
    parser.add_argument("--max-control-pct", type=float, default=1.0)
    parser.add_argument("--min-target-pct", type=float, default=0.05)
    parser.add_argument("--min-target-control-fold", type=float, default=5.0)
    args = parser.parse_args()

    html_path, template_path, summaries = build_dashboard(
        args.config,
        panel=args.panel,
        max_control_pct=args.max_control_pct,
        min_target_pct=args.min_target_pct,
        min_target_control_fold=args.min_target_control_fold,
    )
    candidates = [s.threshold for s in summaries if s.candidate]
    print(f"Wrote manual threshold review dashboard: {html_path}")
    print(f"Wrote sign-off template: {template_path}")
    print("Candidate thresholds:", ", ".join(_threshold_label(v) for v in candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
