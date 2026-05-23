from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import _require_sitk, load_volume


DEFAULT_LABEL_MAP = {
    "DTI": None,
    "DWI": None,
    "FLAIR": "flair",
    "OTHER": None,
    "T1": "t1",
    "T1C": "t1c",
    "T1CE": "t1c",
    "T1GD": "t1c",
    "T1POST": "t1c",
    "T2": "t2",
}


@dataclass(frozen=True)
class SequencePrediction:
    source_path: str
    label: str
    modality: str | None
    confidence: float
    votes: dict[str, int]


def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    arr = slice_2d.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    values = arr[finite]
    lo, hi = np.percentile(values, (1.0, 99.0))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def export_volume_slices(
    volume_path: str | Path,
    output_dir: str | Path,
    num_slices: int = 9,
) -> list[Path]:
    """Export representative axial slices for MRISeqClassifier's 2D image toolkit."""

    _require_sitk()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    volume, _ = load_volume(volume_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume for sequence classification: {volume_path}")

    nonzero_z = np.where(np.any(volume != 0, axis=(1, 2)))[0]
    if nonzero_z.size:
        start, stop = int(nonzero_z.min()), int(nonzero_z.max())
    else:
        start, stop = 0, volume.shape[0] - 1
    if start == stop:
        indices = np.array([start])
    else:
        indices = np.linspace(start, stop, num=min(num_slices, stop - start + 1), dtype=int)

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency belongs to MRISeqClassifier
        raise ImportError("Pillow is required to export classifier input slices") from exc

    written: list[Path] = []
    stem = Path(volume_path).name.replace(".nii.gz", "").replace(".nii", "").replace(".mha", "")
    for index in indices:
        image = Image.fromarray(_normalize_slice(volume[index]))
        path = output / f"{stem}_z{int(index):03d}.jpg"
        image.save(path)
        written.append(path)
    return written


def _read_toolkit_votes(result_csv: Path) -> list[str]:
    with result_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if "vote" not in (reader.fieldnames or []):
            raise ValueError(f"MRISeqClassifier result is missing a vote column: {result_csv}")
        return [row["vote"].strip() for row in reader if row.get("vote")]


def classify_volume_sequence(
    volume_path: str | Path,
    classifier_repo: str | Path,
    python_executable: str = "python",
    num_slices: int = 9,
    label_map: dict[str, str | None] | None = None,
) -> SequencePrediction:
    """Classify one MRI volume with a local MRISeqClassifier checkout.

    The upstream toolkit classifies 2D images and writes `result.csv`. We export
    representative slices for one volume, run `05_toolkit.py`, and majority-vote
    its slice-level predictions into a volume-level modality prediction.
    """

    repo = Path(classifier_repo)
    toolkit = repo / "05_toolkit.py"
    if not toolkit.exists():
        raise FileNotFoundError(f"Could not find MRISeqClassifier toolkit script: {toolkit}")
    if not (repo / "02_models" / "best_model").exists():
        raise FileNotFoundError(
            "MRISeqClassifier best models were not found. Download them into "
            f"{repo / '02_models' / 'best_model'} as described by the upstream README."
        )

    label_map = label_map or DEFAULT_LABEL_MAP
    with tempfile.TemporaryDirectory(prefix="longitumor_mriseq_") as tmp:
        image_dir = Path(tmp) / "slices"
        export_volume_slices(volume_path, image_dir, num_slices=num_slices)
        result_csv = repo / "result.csv"
        if result_csv.exists():
            result_csv.unlink()
        subprocess.run(
            [python_executable, str(toolkit), "--path", str(image_dir)],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
        if not result_csv.exists():
            raise RuntimeError("MRISeqClassifier completed but did not create result.csv")
        local_result = Path(tmp) / "result.csv"
        shutil.copy2(result_csv, local_result)
        votes = _read_toolkit_votes(local_result)

    if not votes:
        raise RuntimeError(f"MRISeqClassifier produced no predictions for {volume_path}")
    counts = Counter(votes)
    label, count = counts.most_common(1)[0]
    normalized = label.upper().replace("-", "").replace("_", "").replace(" ", "")
    modality = label_map.get(normalized)
    return SequencePrediction(
        source_path=str(volume_path),
        label=label,
        modality=modality,
        confidence=count / len(votes),
        votes=dict(counts),
    )
