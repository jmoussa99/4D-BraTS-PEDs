from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

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
DATE_RE = re.compile(r"(?:19|20)\d{2}[-_]?\d{2}[-_]?\d{2}|(?:19|20)\d{6}")
DAYS_RE = re.compile(r"^(\d+)d(?:[_\-\s]|$)", re.IGNORECASE)
SERIES_DIR_RE = re.compile(r"^\d{1,4}\s+[-_]\s+")
VISIT_RE = re.compile(r"(?:^|[_\-\s])(visit|timepoint|tp|scan|study|ses|session|fu|followup)[_\-\s]*([a-z0-9]+)", re.IGNORECASE)
SEGMENTATION_TOKENS = {
    "seg",
    "segs",
    "segmentation",
    "segmentations",
    "mask",
    "masks",
    "label",
    "labels",
    "annotation",
    "annotations",
    "truth",
    "manual",
    "gt",
}
MODALITY_ALIASES = {
    "flair": {
        "flair",
        "tirm",
        "darkfluid",
        "dark_fluid",
        "dark-fluid",
        "t2flair",
        "t2_flair",
        "t2-flair",
        "fluidattenuatedinversionrecovery",
        "fluid_attenuated_inversion_recovery",
        "fluid-attenuated-inversion-recovery",
    },
    "t1c": {
        "t1c",
        "t1ce",
        "t1gd",
        "t1_gd",
        "t1-gd",
        "t1gad",
        "t1_gad",
        "t1-gad",
        "t1post",
        "t1_post",
        "t1-post",
        "t1wpost",
        "t1w_post",
        "t1w-post",
        "t1weightedpost",
        "t1weighted_post",
        "t1weighted-post",
        "postcontrast",
        "post_contrast",
        "post-contrast",
        "postgad",
        "post_gad",
        "post-gad",
        "ce",
        "cetr1",
        "cet1",
        "ce_t1",
        "ce-t1",
        "contrast",
        "enhanced",
    },
    "t1": {
        "t1",
        "t1w",
        "t1weighted",
        "t1_weighted",
        "t1-weighted",
        "t1pre",
        "t1_pre",
        "t1-pre",
        "t1wpre",
        "t1w_pre",
        "t1w-pre",
        "precontrast",
        "pre_contrast",
        "pre-contrast",
        "native",
    },
    "t2": {"t2", "t2w", "t2weighted", "t2_weighted", "t2-weighted"},
}


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


def _is_metadata_dir(path: Path) -> bool:
    return path.name.startswith(".") or path.name == "__MACOSX"


def _iter_image_files(path: Path, recursive: bool = False) -> Iterable[Path]:
    items = path.rglob("*") if recursive else path.iterdir()
    return (item for item in items if _is_image_file(item))


def _strip_image_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in (".nii.gz", ".nii", ".mha"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _tokens(path: Path) -> set[str]:
    stem = _strip_image_suffix(path.name).lower()
    pieces = [piece for piece in re.split(r"[^a-z0-9]+", stem) if piece]
    tokens = set(pieces)
    tokens.add("".join(pieces))
    for left, right in zip(pieces, pieces[1:]):
        tokens.add(left + right)
        tokens.add(f"{left}_{right}")
        tokens.add(f"{left}-{right}")
    return tokens


def _looks_like_segmentation_name(path: Path) -> bool:
    toks = _tokens(path)
    preferred = {token.lower() for token in PREFERRED_SEG_TOKENS if len(token) > 2}
    return bool(toks & SEGMENTATION_TOKENS or any(token in path.name.lower() for token in preferred))


def infer_modality(path: Path) -> str | None:
    """Infer MRI modality from a NIfTI/MHA filename.

    Explicit contrast names win over BraTS/nnU-Net numeric suffixes. This keeps
    arbitrary clinical filenames usable while preserving `_0000.._0003` support.
    """

    toks = _tokens(path)
    if toks & SEGMENTATION_TOKENS:
        return None
    if (toks & MODALITY_ALIASES["t1"]) and {"post", "gad", "gd", "contrast", "enhanced"} & toks:
        return "t1c"
    for modality in ("flair", "t1c", "t2", "t1"):
        if toks & MODALITY_ALIASES[modality]:
            return modality
    match = MODALITY_SUFFIX_RE.search(path.name)
    if match:
        return MODALITY_NAMES[int(match.group(1))]
    return None


def _modality_score(path: Path, modality: str) -> tuple[int, int, str]:
    toks = _tokens(path)
    exact = int(modality in toks)
    alias_hits = len(toks & MODALITY_ALIASES[modality])
    numeric = int(MODALITY_SUFFIX_RE.search(path.name) is not None)
    return exact + alias_hits + numeric, path.stat().st_size, path.name


ModalityClassifier = Callable[[Path], tuple[str, float]]


def discover_modalities(
    case_dir: Path,
    modality_classifier: ModalityClassifier | None = None,
    classifier_threshold: float = 0.70,
) -> tuple[str | None, ...]:
    """Return modality paths ordered as T1, T2, T1c, FLAIR."""

    paths: list[str | None] = [None] * len(MODALITY_NAMES)
    candidates: dict[str, list[Path]] = {name: [] for name in MODALITY_NAMES}
    classifier_scores: dict[Path, float] = {}
    for item in _iter_image_files(case_dir, recursive=True):
        modality = infer_modality(item)
        if modality is None and modality_classifier is not None:
            try:
                if _looks_like_label_volume(item):
                    continue
            except Exception:
                pass
            predicted_modality, confidence = modality_classifier(item)
            if confidence >= classifier_threshold and predicted_modality in candidates:
                modality = predicted_modality
                classifier_scores[item] = confidence
        if modality is None:
            continue
        candidates[modality].append(item)
    for idx, modality in enumerate(MODALITY_NAMES):
        if candidates[modality]:
            paths[idx] = str(
                sorted(
                    candidates[modality],
                    key=lambda p: (classifier_scores.get(p, 0.0), *_modality_score(p, modality)),
                    reverse=True,
                )[0]
            )
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


def find_segmentation_file(case_dir: Path, excluded_paths: Sequence[str | None] = ()) -> Path | None:
    excluded = {str(Path(path)) for path in excluded_paths if path}
    candidates = [
        p
        for p in _iter_image_files(case_dir, recursive=True)
        if _is_image_file(p)
        and str(p) not in excluded
        and infer_modality(p) is None
        and _looks_like_segmentation_name(p)
    ]
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        token_score = sum(1 for token in PREFERRED_SEG_TOKENS if token in path.name)
        return token_score, -path.stat().st_size, path.name

    candidates.sort(key=score, reverse=True)
    for candidate in candidates:
        try:
            if _looks_like_label_volume(candidate):
                return candidate
        except ImportError:
            continue
    if sitk is None:
        return None
    for candidate in candidates:
        if candidate.name.lower().endswith(SEG_EXTENSIONS):
            return candidate
    return None


def _contains_images(path: Path) -> bool:
    if not path.is_dir() or _is_metadata_dir(path):
        return False
    return any(_iter_image_files(path))


def _contains_modalities(path: Path, modality_classifier: ModalityClassifier | None = None) -> bool:
    if not _contains_images(path):
        return False
    if modality_classifier is not None:
        return True
    return any(infer_modality(item) is not None for item in _iter_image_files(path))


def _looks_like_visit_dir(path: Path, modality_classifier: ModalityClassifier | None = None) -> bool:
    if _contains_modalities(path, modality_classifier):
        return True
    if not path.is_dir() or _is_metadata_dir(path):
        return False
    return any(
        child.is_dir()
        and not _is_metadata_dir(child)
        and SERIES_DIR_RE.search(child.name)
        and _contains_modalities(child, modality_classifier)
        for child in path.iterdir()
    )


def _visit_sort_key(path: Path) -> tuple[int, str]:
    text = path.name.lower()
    date_match = DATE_RE.search(text)
    if date_match:
        digits = re.sub(r"\D", "", date_match.group(0))
        return int(digits), path.name
    days_match = DAYS_RE.search(text)
    if days_match:
        return int(days_match.group(1)), path.name
    visit_match = VISIT_RE.search(text)
    if visit_match:
        raw = visit_match.group(2)
        if raw.isdigit():
            return int(raw), path.name
        return 10_000 + sum(ord(char) for char in raw), path.name
    return 1_000_000, path.name


def _visit_id(path: Path, patient_dir: Path) -> str:
    if path == patient_dir:
        return "baseline"
    return path.relative_to(patient_dir).as_posix().replace("/", "__")


def _visit_delta_months(visit_dir: Path, ordered_visit_dirs: Sequence[Path]) -> float:
    index = ordered_visit_dirs.index(visit_dir)
    key = _visit_sort_key(visit_dir)[0]
    first_key = _visit_sort_key(ordered_visit_dirs[0])[0]
    if 19_000_000 <= key <= 20_999_999 and 19_000_000 <= first_key <= 20_999_999:
        from datetime import datetime

        current = datetime.strptime(str(key), "%Y%m%d")
        first = datetime.strptime(str(first_key), "%Y%m%d")
        return max(0.0, (current - first).days / 30.4375)
    if key < 1_000_000 and first_key < 1_000_000:
        return max(0.0, (key - first_key) / 30.4375)
    return float(index)


def _discover_visit_dirs(patient_dir: Path, modality_classifier: ModalityClassifier | None = None) -> list[Path]:
    visit_dirs = [patient_dir] if _looks_like_visit_dir(patient_dir, modality_classifier) else []
    for child in sorted(patient_dir.rglob("*")):
        if child == patient_dir or not child.is_dir() or _is_metadata_dir(child):
            continue
        if any(parent in visit_dirs for parent in child.parents):
            continue
        if _looks_like_visit_dir(child, modality_classifier):
            visit_dirs.append(child)
    return sorted(set(visit_dirs), key=_visit_sort_key)


def _patient_dirs(data_dir: Path, modality_classifier: ModalityClassifier | None = None) -> list[Path]:
    dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir() and not _is_metadata_dir(p)]
    if len(dirs) == 1 and not _looks_like_visit_dir(dirs[0], modality_classifier):
        nested = [p for p in sorted(dirs[0].iterdir()) if p.is_dir() and not _is_metadata_dir(p)]
        if nested and any(any(visit != p for visit in _discover_visit_dirs(p, modality_classifier)) for p in nested):
            return nested
    return dirs


def discover_cases(
    data_dir: Path,
    modality_classifier: ModalityClassifier | None = None,
    classifier_threshold: float = 0.70,
) -> list[VisitRecord]:
    records: list[VisitRecord] = []
    patient_dirs = _patient_dirs(data_dir, modality_classifier)
    if _contains_modalities(data_dir, modality_classifier):
        patient_dirs.insert(0, data_dir)
    for patient_dir in patient_dirs:
        visit_dirs = _discover_visit_dirs(patient_dir, modality_classifier)
        if not visit_dirs:
            continue
        previous_mask_path: str | None = None
        for visit_dir in visit_dirs:
            modalities = discover_modalities(
                visit_dir,
                modality_classifier=modality_classifier,
                classifier_threshold=classifier_threshold,
            )
            if not any(modalities):
                continue
            mask_path = find_segmentation_file(visit_dir, excluded_paths=modalities)
            records.append(
                VisitRecord(
                    patient_id=patient_dir.name,
                    visit_id=_visit_id(visit_dir, patient_dir),
                    delta_t=_visit_delta_months(visit_dir, visit_dirs),
                    modalities=modalities,
                    mask_path=str(mask_path) if mask_path is not None else None,
                    previous_mask_path=previous_mask_path,
                )
            )
            if mask_path is not None:
                previous_mask_path = str(mask_path)
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
