from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from longitumor.data import discover_cases, discover_modalities, infer_modality, labels_to_channels, random_patch_slices
from longitumor.models import LongiTumorMamba, LongiTumorMambaConfig
from longitumor.training import DiceBCELoss


def test_labels_to_channels() -> None:
    labels = torch.tensor([[[1, 2], [3, 4]]]).numpy()
    channels = labels_to_channels(labels)
    assert channels.shape == (4, 1, 2, 2)
    assert channels.sum() == 4


def test_random_patch_slices() -> None:
    slices = random_patch_slices((20, 30, 40), (8, 12, 16))
    assert len(slices) == 3
    assert all((sl.stop or 0) - (sl.start or 0) <= size for sl, size in zip(slices, (8, 12, 16)))


def test_infer_modality_from_clinical_filenames(tmp_path) -> None:
    expected = {
        "patientA_baseline_T1w_pre.nii.gz": "t1",
        "patientA_baseline_T2_axial.nii.gz": "t2",
        "patientA_baseline_T1w_POST_GAD.nii.gz": "t1c",
        "patientA_baseline_FLAIR.nii.gz": "flair",
        "patientA_baseline_segmentation.nii.gz": None,
    }
    for name, modality in expected.items():
        path = tmp_path / name
        path.touch()
        assert infer_modality(path) == modality


def test_discover_modalities_from_named_files(tmp_path) -> None:
    for name in (
        "study_native_T1w.nii.gz",
        "study_T2w.nii.gz",
        "study_T1ce.nii.gz",
        "study_t2_flair.nii.gz",
    ):
        (tmp_path / name).touch()
    assert tuple(path and Path(path).name for path in discover_modalities(tmp_path)) == (
        "study_native_T1w.nii.gz",
        "study_T2w.nii.gz",
        "study_T1ce.nii.gz",
        "study_t2_flair.nii.gz",
    )


def test_discover_cases_from_nested_visits(tmp_path) -> None:
    patient = tmp_path / "patient_001"
    baseline = patient / "2024-01-01"
    followup = patient / "2024-03-01"
    baseline.mkdir(parents=True)
    followup.mkdir(parents=True)
    for visit in (baseline, followup):
        for name in ("T1w_pre.nii.gz", "T2w.nii.gz", "T1w_post.nii.gz", "FLAIR.nii.gz"):
            (visit / name).touch()
    records = discover_cases(tmp_path)
    assert [record.visit_id for record in records] == ["2024-01-01", "2024-03-01"]
    assert records[0].delta_t == 0.0
    assert records[1].delta_t > 1.0
    assert all(all(record.modalities) for record in records)


def test_discover_cases_with_content_classifier_for_anonymous_files(tmp_path) -> None:
    visit = tmp_path / "patient_001" / "visit_01"
    visit.mkdir(parents=True)
    labels = {
        "anon_a.nii.gz": "t1",
        "anon_b.nii.gz": "t2",
        "anon_c.nii.gz": "t1c",
        "anon_d.nii.gz": "flair",
    }
    for name in labels:
        (visit / name).touch()

    def classifier(path: Path) -> tuple[str, float]:
        return labels[path.name], 0.99

    records = discover_cases(tmp_path, modality_classifier=classifier)
    assert len(records) == 1
    assert tuple(Path(path).name if path else None for path in records[0].modalities) == (
        "anon_a.nii.gz",
        "anon_b.nii.gz",
        "anon_c.nii.gz",
        "anon_d.nii.gz",
    )


def test_longitumor_forward_backward() -> None:
    model = LongiTumorMamba(LongiTumorMambaConfig(base_channels=4, embedding_dim=16, use_mamba=False))
    x = torch.randn(1, 2, 4, 16, 16, 16)
    availability = torch.ones(1, 2, 4)
    target = torch.rand(1, 2, 4, 16, 16, 16).round()
    output = model(x, availability=availability)
    assert output.logits.shape == target.shape
    assert output.trajectory_embedding.shape == (1, 2, 16)
    loss = DiceBCELoss()(output.logits, target)
    loss.backward()
    assert torch.isfinite(loss)
