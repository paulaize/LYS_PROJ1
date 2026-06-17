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


def test_all_template_panel_channels_require_confirmed_igg_fitc_index():
    cfg = load_config("config/animals/TEMPLATE.yml", repo_root=REPO)
    for panel in ("A", "B"):
        with pytest.raises(ValueError, match="igg_fitc_channel_index is unset"):
            cfg.panel_igg_fitc_channel_index(panel)


def test_template_panel_thresholds_require_approved_value():
    cfg = load_config("config/animals/TEMPLATE.yml", repo_root=REPO)
    for panel in ("A", "B"):
        with pytest.raises(ValueError, match="IgG-FITC positive threshold is unset"):
            cfg.panel_igg_fitc_threshold(panel)


def test_real_v1_animal_config_records_raw_and_derived_mri_paths():
    cfg = load_config("config/animals/BD_08_5D.yml", repo_root=REPO)
    assert cfg.animal_id == "BD_08_5D"
    assert cfg.timepoint == "5d"
    assert cfg.bruker_study == REPO / "data/5D/MRI/20251123_175503_JD_BD_08_2_D5_MRI_D5"
    assert cfg.t2_scan_id == 2
    assert cfg.t2_reco_id == 1
    assert cfg.t2_nifti == REPO / "work/BD_08_5D/mri/t2_scan2.nii.gz"
    assert cfg.animal["mri"]["lesion_side"] == "image_right"


def test_real_v1_animal_has_confirmed_ihc_channels_but_unset_thresholds():
    cfg = load_config("config/animals/BD_08_5D.yml", repo_root=REPO)
    for panel in ("A", "B"):
        assert cfg.panel_igg_fitc_channel_index(panel) == 1
        with pytest.raises(ValueError):
            cfg.panel_igg_fitc_threshold(panel)


def test_real_v1_animal_resolves_anti_human_igg_specificity():
    cfg = load_config("config/animals/BD_08_5D.yml", repo_root=REPO)
    flags = cfg.animal["interpretation_flags"]
    assert flags["anti_igg_specificity_resolved"] is True
    assert flags["fitc_specific_to_lys241"] is True
    assert flags["specificity_source"] == "confirmed_anti_human_IgG (Paul, 2026-06-17)"
    assert flags["fitc_igg_specificity"] == "anti_human_confirmed"


def test_real_v1_animal_splits_panel_spectral_bleedthrough():
    cfg = load_config("config/animals/BD_08_5D.yml", repo_root=REPO)
    assert cfg.panel_config("A")["fitc_spectral_bleedthrough"] == "pending_neurotrace_emission"
    assert cfg.panel_config("B")["fitc_spectral_bleedthrough"] == "none"


def test_real_v1_animal_records_qupath_series_layout():
    cfg = load_config("config/animals/BD_08_5D.yml", repo_root=REPO)
    qupath = cfg.animal["ihc"]["qupath"]
    assert qupath["tissue_annotation_name"] == "tissue_v1"
    assert qupath["combined_csv_mode"] == "overwrite"
    assert qupath["label_series_index"] == 0
    assert qupath["overview_series_index"] == 1
    assert qupath["section_series_indices"] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert qupath["slide_row_major_section_ids"] == [
        "section_04",
        "section_03",
        "section_02",
        "section_01",
        "section_08",
        "section_07",
        "section_06",
        "section_05",
    ]
