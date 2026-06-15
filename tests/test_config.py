"""Config loader tests."""
from pathlib import Path

import pytest

from src.config import load_config

REPO = Path(__file__).resolve().parents[1]


def test_template_config_loads():
    cfg = load_config("config/animals/TEMPLATE.yml", repo_root=REPO)
    assert cfg.animal_id == "TEMPLATE"
    assert cfg.expected_spacing_mm == (0.07, 0.07, 0.5)
    assert cfg.version


def test_require_raises_on_unset_todo():
    cfg = load_config("config/animals/TEMPLATE.yml", repo_root=REPO)
    with pytest.raises(ValueError):
        cfg.require("ihc.igg_fitc_positive_threshold")


def test_require_raises_on_missing_key():
    cfg = load_config("config/animals/TEMPLATE.yml", repo_root=REPO)
    with pytest.raises(KeyError):
        cfg.require("nonexistent.key.path")


def test_panel_channel_requires_confirmed_igg_fitc_index():
    cfg = load_config("config/animals/TEMPLATE.yml", repo_root=REPO)
    with pytest.raises(ValueError):
        cfg.panel_igg_fitc_channel_index("A")
