import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_dashboard():
    path = REPO / "scripts/build_ihc_threshold_review_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_ihc_threshold_review_dashboard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_threshold_dashboard_filters_to_plausible_candidates():
    dashboard = _load_dashboard()
    example = {
        100: (15.45, 15.19),
        250: (3.99, 0.35),
        500: (0.31, 0.003),
        1000: (0.005, 0.00016),
    }
    rows = []
    for threshold, (target, control) in example.items():
        rows.append(
            {
                "source_role": "target",
                "threshold": str(threshold),
                "igg_fitc_pct_positive_area": str(target),
            }
        )
        rows.append(
            {
                "source_role": "no_lys241_control",
                "threshold": str(threshold),
                "igg_fitc_pct_positive_area": str(control),
            }
        )

    summaries = dashboard.summarize_thresholds(rows)
    candidates = {int(s.threshold) for s in summaries if s.candidate}

    assert candidates == {250, 500}
    rejected = {int(s.threshold): s.reason for s in summaries if not s.candidate}
    assert "control mean" in rejected[100]
    assert "target mean" in rejected[1000]
