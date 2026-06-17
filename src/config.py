"""Load and validate pipeline + per-animal config.

Everything the pipeline needs comes from YAML: paths, channels, thresholds, and
interpretation flags. Keep hardcoded domain facts out of the analysis code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return data


def _get_dotted(mapping: dict[str, Any], dotted_key: str) -> Any:
    node: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing required config key: {dotted_key}")
        node = node[part]
    return node


@dataclass
class Config:
    """Merged view of config/pipeline.yml + one config/animals/<id>.yml."""

    pipeline: dict[str, Any]
    animal: dict[str, Any]
    repo_root: Path = field(default_factory=lambda: Path.cwd())

    @property
    def animal_id(self) -> str:
        return str(self.animal.get("animal_id", "UNKNOWN"))

    @property
    def timepoint(self) -> str | None:
        value = self.animal.get("timepoint")
        return None if value is None else str(value)

    @property
    def version(self) -> str:
        return str(self.pipeline.get("version", "unknown"))

    @property
    def has_mri(self) -> bool:
        return bool(self.animal.get("mri", {}).get("has_mri", False))

    @property
    def t2_nifti(self) -> Path | None:
        p = self.animal.get("mri", {}).get("t2_nifti")
        return self.resolve_input_path(p) if p else None

    @property
    def bruker_study(self) -> Path | None:
        p = self.animal.get("mri", {}).get("bruker_study")
        return self.resolve_input_path(p) if p else None

    @property
    def t2_scan_id(self) -> int:
        value = self.animal.get("mri", {}).get("t2_scan_id")
        if value is None:
            value = self.pipeline.get("mri", {}).get("lesion_scan", {}).get("scan_number", 2)
        return int(value)

    @property
    def t2_reco_id(self) -> int:
        value = self.animal.get("mri", {}).get("t2_reco_id", 1)
        return int(value)

    @property
    def expected_spacing_mm(self) -> tuple[float, float, float]:
        s = self.pipeline.get("mri", {}).get("expected_spacing_mm", [0.07, 0.07, 0.5])
        return (float(s[0]), float(s[1]), float(s[2]))

    @property
    def spacing_tolerance(self) -> float:
        return float(self.pipeline.get("mri", {}).get("spacing_tolerance", 0.02))

    @property
    def reviewer(self) -> str | None:
        value = self.pipeline.get("mri", {}).get("edit", {}).get("reviewer")
        return None if value in (None, "", "TODO") else str(value)

    def resolve_input_path(self, value: str | Path | None) -> Path | None:
        """Resolve configured input paths without creating or mutating them.

        Relative paths are interpreted relative to repo_root. Absolute paths,
        including `/Volumes/...`, are allowed as read-only inputs.
        """
        if value in (None, "", "TODO"):
            return None
        p = Path(value)
        return p if p.is_absolute() else self.repo_root / p

    def work_dir(self) -> Path:
        root = Path(self.pipeline.get("paths", {}).get("work_root", "work"))
        d = self.repo_root / root / self.animal_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def outputs_dir(self) -> Path:
        paths = self.pipeline.get("paths", {})
        root = Path(paths.get("outputs_root", paths.get("output_root", "outputs")))
        d = self.repo_root / root / self.animal_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def require(self, dotted_key: str, *, source: str = "pipeline") -> Any:
        """Fetch a required value; raise if it is unset/TODO.

        Example: cfg.require("ihc.igg_fitc_positive_threshold")
        """
        mapping = self.pipeline if source == "pipeline" else self.animal
        node = _get_dotted(mapping, dotted_key)
        if node is None or node == "TODO":
            raise ValueError(
                f"Config key '{dotted_key}' in {source} is still unset (TODO). "
                "Fill it before running this stage — see AGENTS.md §12."
            )
        return node

    def panel_config(self, panel: str) -> dict[str, Any]:
        panels = self.animal.get("ihc", {}).get("panels", {})
        if panel not in panels:
            raise KeyError(f"Animal {self.animal_id} has no IHC panel {panel!r} in config")
        cfg = panels[panel]
        if not isinstance(cfg, dict):
            raise ValueError(f"IHC panel {panel!r} config must be a mapping")
        return cfg

    def panel_igg_fitc_channel_index(self, panel: str) -> int:
        cfg = self.panel_config(panel)
        idx = cfg.get("igg_fitc_channel_index")
        if idx is None:
            # Fallback: find marker in channel_map after Paul confirms it.
            for key, marker in (cfg.get("channel_map") or {}).items():
                if str(marker).strip().lower() in {"igg-fitc", "igg_fitc", "anti-igg-fitc"}:
                    idx = key
                    break
        if idx is None:
            raise ValueError(
                f"IHC panel {panel}: igg_fitc_channel_index is unset. Confirm channel order in "
                "QuPath/Bio-Formats and fill config/animals/<id>.yml."
            )
        return int(idx)

    def panel_igg_fitc_threshold(self, panel: str) -> float:
        cfg = self.panel_config(panel)
        value = cfg.get("igg_fitc_positive_threshold")
        if value is None:
            value = self.pipeline.get("ihc", {}).get("igg_fitc_positive_threshold")
        if value is None:
            raise ValueError(
                f"IHC panel {panel}: IgG-FITC positive threshold is unset. Tune/confirm it "
                "before trusting % positive area."
            )
        return float(value)


def load_config(
    animal_config: str | Path,
    pipeline_config: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Config:
    """Load the global + per-animal config into one Config object."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    animal_path = Path(animal_config)
    if not animal_path.is_absolute():
        animal_path = repo_root / animal_path
    pipeline_path = (
        Path(pipeline_config) if pipeline_config else repo_root / "config" / "pipeline.yml"
    )
    if not pipeline_path.is_absolute():
        pipeline_path = repo_root / pipeline_path

    return Config(
        pipeline=_load_yaml(pipeline_path),
        animal=_load_yaml(animal_path),
        repo_root=repo_root,
    )
