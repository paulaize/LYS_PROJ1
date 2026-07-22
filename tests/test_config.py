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
    assert cfg.panel_config("A")["neurotrace_emission_confirmed"] is True
    assert cfg.panel_config("A")["fitc_spectral_bleedthrough"] == "none"
    assert cfg.panel_config("B")["fitc_spectral_bleedthrough"] == "none"


def test_real_v1_animal_records_panel_specific_section_selection():
    cfg = load_config("config/animals/BD_08_5D.yml", repo_root=REPO)
    assert cfg.panel_config("A")["section_selection"]["selected_section_ids"] == [
        "section_01",
        "section_03",
        "section_06",
    ]
    assert cfg.panel_config("B")["section_selection"]["selected_section_ids"] == [
        "section_05",
        "section_06",
        "section_08",
    ]


def test_control_animal_config_records_ihc_without_mri():
    cfg = load_config("config/animals/C6S5.yml", repo_root=REPO)
    assert cfg.animal_id == "C6S5"
    assert cfg.has_mri is False
    assert cfg.animal["group"] == "control"
    for panel in ("A", "B"):
        assert cfg.panel_igg_fitc_channel_index(panel) == 1
        with pytest.raises(ValueError):
            cfg.panel_igg_fitc_threshold(panel)


def test_control_config_records_panel_a_selection_and_panel_b_available_unreviewed():
    cfg = load_config("config/animals/C6S5.yml", repo_root=REPO)
    assert cfg.panel_config("A")["section_selection"]["selected_section_ids"] == [
        "section_05",
        "section_06",
        "section_07",
    ]
    panel_b = cfg.panel_config("B")
    assert panel_b["control_status"] == "available_pending_visual_qc"
    assert panel_b["exclude_from_threshold_calibration"] is False
    assert cfg.panel_config("B")["section_selection"]["selected_section_ids"] == []
    assert cfg.panel_config("B")["section_selection"]["excluded_section_ids"] == []
    assert (
        cfg.panel_config("B")["section_selection"]["exploratory_control_selection_status"]
        == "technical_open_only_unreviewed"
    )


def test_real_ihc_configs_record_complete_olympus_source_bundles():
    for animal_id in ("BD_08_5D", "C6S5"):
        cfg = load_config(f"config/animals/{animal_id}.yml", repo_root=REPO)
        for panel in ("A", "B"):
            panel_cfg = cfg.panel_config(panel)
            bundles = panel_cfg["source_bundles"]
            assert len(bundles) == 1
            assert bundles[0]["vsi_file"] == panel_cfg["vsi_files"][0]
            assert bundles[0]["companion_dir"].endswith("_")


def test_real_v1_animal_records_qupath_series_layout():
    cfg = load_config("config/animals/BD_08_5D.yml", repo_root=REPO)
    ihc = cfg.animal["ihc"]
    assert ihc["section_thickness_um"] == 10
    assert ihc["section_spacing_um"] is None
    assert ihc["panel_section_matching"] == "unconfirmed_do_not_match_by_section_id"
    assert ihc["ipsilateral_side_in_image"] == "right"
    assert ihc["orientation_qc_status"] == "pending_visual_qc"
    qupath = cfg.animal["ihc"]["qupath"]
    assert qupath["tissue_annotation_name"] == "tissue_v1"
    assert qupath["artifact_exclusion_annotation_names"] == ["artifact_exclude"]
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


def test_control_config_records_section_metadata_without_lesion_side():
    cfg = load_config("config/animals/C6S5.yml", repo_root=REPO)
    ihc = cfg.animal["ihc"]
    assert ihc["section_thickness_um"] == 10
    assert ihc["section_spacing_um"] is None
    assert ihc["panel_section_matching"] == "unconfirmed_do_not_match_by_section_id"
    assert ihc["ipsilateral_side_in_image"] is None
    assert ihc["orientation_qc_status"] == "not_applicable_control"
    assert ihc["qupath"]["artifact_exclusion_annotation_names"] == ["artifact_exclude"]
