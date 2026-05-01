from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    sitk = None
    _sitk_import_error = exc
else:
    _sitk_import_error = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - lets discovery utilities run without torch
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


MODALITY_NAMES = ("t1", "t2", "t1c", "flair")
MODALITY_SUFFIX_RE = re.compile(r"_000([0-3])([_\s][^/]*)?\.nii(\.gz)?$", re.IGNORECASE)
SEG_EXTENSIONS = (".nii", ".nii.gz", ".mha")
PREFERRED_SEG_TOKENS = ("REVISED", "UPDATED", "v2", "V2", "my_", "MY_", "ens", "ENS", "E-", "E_")
MAX_SEG_UNIQUE_VALUES = 16


@dataclass(frozen=True)
class VisitRecord:
    patient_id: str
    visit_id: str
    delta_t: float
    modalities: tuple[str | None, ...]
    mask_path: str | None
    previous_mask_path: str | None = None


def _require_sitk() -> None:
    if sitk is None:
        raise ImportError("SimpleITK is required for MRI loading") from _sitk_import_error


def _is_image_file(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and not path.name.startswith(".") and name.endswith(SEG_EXTENSIONS)


def discover_modalities(case_dir: Path) -> tuple[str | None, ...]:
    """Return modality paths ordered as T1, T2, T1c, FLAIR.

    The pedBRATS folders in this workspace use `_0000` through `_0003` suffixes.
    We keep the architecture order stable (`T1`, `T2`, `T1c`, `FLAIR`) while
    accepting the numeric filenames present on disk.
    """

    paths: list[str | None] = [None] * len(MODALITY_NAMES)
    for item in case_dir.iterdir():
        if not _is_image_file(item):
            continue
        match = MODALITY_SUFFIX_RE.search(item.name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx < len(paths):
            paths[idx] = str(item)
    return tuple(paths)


def _looks_like_label_volume(path: Path) -> bool:
    _require_sitk()
    try:
        arr = sitk.GetArrayViewFromImage(sitk.ReadImage(str(path)))
    except Exception:
        return False
    flat = np.asarray(arr).ravel()
    if flat.size > 5_000_000:
        flat = flat[:: max(1, flat.size // 5_000_000)]
    uniq = np.unique(flat)
    if uniq.size > MAX_SEG_UNIQUE_VALUES:
        return False
    if not np.all(uniq == uniq.astype(np.int64)):
        return False
    return bool(uniq.min() >= 0 and uniq.max() <= 20)


def find_segmentation_file(case_dir: Path) -> Path | None:
    candidates = [
        p
        for p in case_dir.iterdir()
        if _is_image_file(p) and MODALITY_SUFFIX_RE.search(p.name) is None
    ]
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        token_score = sum(1 for token in PREFERRED_SEG_TOKENS if token in path.name)
        return token_score, -path.stat().st_size, path.name

    candidates.sort(key=score, reverse=True)
    for candidate in candidates:
        if _looks_like_label_volume(candidate):
            return candidate
    return candidates[0]


def discover_cases(data_dir: Path) -> list[VisitRecord]:
    records: list[VisitRecord] = []
    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith("."):
            continue
        modalities = discover_modalities(case_dir)
        if not any(modalities):
            continue
        mask_path = find_segmentation_file(case_dir)
        records.append(
            VisitRecord(
                patient_id=case_dir.name,
                visit_id="baseline",
                delta_t=0.0,
                modalities=modalities,
                mask_path=str(mask_path) if mask_path is not None else None,
            )
        )
    return records


def write_manifest(records: Sequence[VisitRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "visit_id", "delta_t", *MODALITY_NAMES, "mask", "previous_mask"]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "patient_id": record.patient_id,
                "visit_id": record.visit_id,
                "delta_t": record.delta_t,
                "mask": record.mask_path or "",
                "previous_mask": record.previous_mask_path or "",
            }
            row.update({name: path or "" for name, path in zip(MODALITY_NAMES, record.modalities)})
            writer.writerow(row)


def read_manifest(path: Path) -> list[VisitRecord]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            records.append(
                VisitRecord(
                    patient_id=row["patient_id"],
                    visit_id=row.get("visit_id", "baseline"),
                    delta_t=float(row.get("delta_t") or 0.0),
                    modalities=tuple(row.get(name) or None for name in MODALITY_NAMES),
                    mask_path=row.get("mask") or None,
                    previous_mask_path=row.get("previous_mask") or None,
                )
            )
    return records


def load_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    _require_sitk()
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = tuple(float(v) for v in image.GetSpacing())
    return array, spacing


def zscore_nonzero(volume: np.ndarray) -> np.ndarray:
    out = volume.astype(np.float32, copy=True)
    mask = out != 0
    if not np.any(mask):
        return out
    mean = float(out[mask].mean())
    std = float(out[mask].std())
    if std < 1e-6:
        out[mask] = 0.0
    else:
        out[mask] = (out[mask] - mean) / std
    return out


def labels_to_channels(label: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Convert integer tumor labels 1..num_classes to channel masks."""

    channels = [(label == cls).astype(np.float32) for cls in range(1, num_classes + 1)]
    return np.stack(channels, axis=0)


def random_patch_slices(
    shape_zyx: Sequence[int],
    patch_size_zyx: Sequence[int],
    foreground: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[slice, slice, slice]:
    rng = rng or np.random.default_rng()
    starts: list[int] = []
    if foreground is not None and foreground.any() and rng.random() < 0.7:
        center = np.array(np.where(foreground)).T[rng.integers(int(foreground.sum()))]
    else:
        center = np.array([rng.integers(dim) for dim in shape_zyx])
    for dim, patch, c in zip(shape_zyx, patch_size_zyx, center):
        if patch >= dim:
            starts.append(0)
        else:
            starts.append(int(np.clip(c - patch // 2, 0, dim - patch)))
    return tuple(slice(start, min(start + patch, dim)) for start, patch, dim in zip(starts, patch_size_zyx, shape_zyx))  # type: ignore[return-value]


class SingleVisitMRIDataset(Dataset):
    def __init__(
        self,
        records: Sequence[VisitRecord],
        patch_size: tuple[int, int, int] | None = (96, 160, 160),
        training: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for SingleVisitMRIDataset")
        self.records = list(records)
        self.patch_size = patch_size
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        volumes: list[np.ndarray] = []
        availability: list[float] = []
        reference_shape: tuple[int, int, int] | None = None
        spacing = (1.0, 1.0, 1.0)
        for path in record.modalities:
            if path:
                volume, spacing = load_volume(path)
                volume = zscore_nonzero(volume)
                reference_shape = volume.shape
                volumes.append(volume)
                availability.append(1.0)
            else:
                if reference_shape is None:
                    raise ValueError(f"Cannot infer missing modality shape for {record.patient_id}")
                volumes.append(np.zeros(reference_shape, dtype=np.float32))
                availability.append(0.0)
        image = np.stack(volumes, axis=0)

        if record.mask_path:
            label, _ = load_volume(record.mask_path)
            target = labels_to_channels(label.astype(np.int16))
            foreground = target.any(axis=0)
        else:
            target = np.zeros((4, *image.shape[1:]), dtype=np.float32)
            foreground = None

        if self.patch_size is not None:
            zsl, ysl, xsl = random_patch_slices(image.shape[1:], self.patch_size, foreground)
            image = image[:, zsl, ysl, xsl]
            target = target[:, zsl, ysl, xsl]

        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target),
            "availability": torch.tensor(availability, dtype=torch.float32),
            "delta_t": torch.tensor(0.0, dtype=torch.float32),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
            "patient_id": record.patient_id,
            "visit_id": record.visit_id,
        }


class LongitudinalMRIDataset(Dataset):
    def __init__(
        self,
        records: Sequence[VisitRecord],
        timepoints: int = 4,
        patch_size: tuple[int, int, int] | None = (96, 160, 160),
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for LongitudinalMRIDataset")
        self.timepoints = timepoints
        self.patch_size = patch_size
        grouped: dict[str, list[VisitRecord]] = {}
        for record in records:
            grouped.setdefault(record.patient_id, []).append(record)
        self.sequences = [sorted(items, key=lambda r: r.visit_id) for items in grouped.values()]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        visits = self.sequences[index][: self.timepoints]
        samples = [SingleVisitMRIDataset([visit], self.patch_size, training=True)[0] for visit in visits]
        return {
            "image": torch.stack([s["image"] for s in samples], dim=0),
            "target": torch.stack([s["target"] for s in samples], dim=0),
            "availability": torch.stack([s["availability"] for s in samples], dim=0),
            "delta_t": torch.tensor([visit.delta_t for visit in visits], dtype=torch.float32),
            "patient_id": visits[0].patient_id,
        }


def collate_single_visit(batch: Iterable[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    items = list(batch)
    return {
        "image": torch.stack([item["image"] for item in items]),  # type: ignore[arg-type]
        "target": torch.stack([item["target"] for item in items]),  # type: ignore[arg-type]
        "availability": torch.stack([item["availability"] for item in items]),  # type: ignore[arg-type]
        "delta_t": torch.stack([item["delta_t"] for item in items]),  # type: ignore[arg-type]
        "spacing": torch.stack([item["spacing"] for item in items]),  # type: ignore[arg-type]
        "patient_id": [str(item["patient_id"]) for item in items],
        "visit_id": [str(item["visit_id"]) for item in items],
    }
