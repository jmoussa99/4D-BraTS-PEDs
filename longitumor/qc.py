from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import MODALITY_NAMES, VisitRecord, looks_like_localizer

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover
    sitk = None
    _sitk_import_error = exc
else:
    _sitk_import_error = None


def _require_sitk() -> None:
    if sitk is None:
        raise ImportError("SimpleITK is required for pseudo-mask QC") from _sitk_import_error


@dataclass(frozen=True)
class MaskQCResult:
    status: str
    reason: str
    volume_ml: float
    foreground_voxels: int
    foreground_slices: int
    slice_fraction: float
    component_count: int
    largest_component_fraction: float


def image_is_segmentation_ready(
    path: str | Path | None,
    min_slices: int = 8,
    min_inplane: int = 64,
) -> tuple[bool, str]:
    if not path:
        return False, "missing"
    image_path = Path(path)
    if looks_like_localizer(image_path):
        return False, "localizer_or_scout"
    if not image_path.exists():
        return False, "missing_file"
    _require_sitk()
    try:
        image = sitk.ReadImage(str(image_path))
    except Exception as exc:
        return False, f"read_error:{exc.__class__.__name__}"
    size = image.GetSize()
    if len(size) < 3:
        return False, "not_3d"
    if min(size[0], size[1]) < min_inplane:
        return False, "small_inplane"
    if size[2] < min_slices:
        return False, "too_few_slices"
    return True, "ok"


def modality_qc_reasons(
    record: VisitRecord,
    min_modalities: int = 2,
    min_slices: int = 8,
    min_inplane: int = 64,
) -> list[str]:
    reasons: list[str] = []
    usable = 0
    for modality, path in zip(MODALITY_NAMES, record.modalities):
        ok, reason = image_is_segmentation_ready(path, min_slices=min_slices, min_inplane=min_inplane)
        if ok:
            usable += 1
        elif path:
            reasons.append(f"{modality}:{reason}")
    if usable < min_modalities:
        reasons.append(f"usable_modalities<{min_modalities}")
    return reasons


def _component_stats(mask: np.ndarray) -> tuple[int, float]:
    _require_sitk()
    binary = sitk.GetImageFromArray(mask.astype(np.uint8))
    components = sitk.ConnectedComponent(binary)
    relabeled = sitk.RelabelComponent(components, sortByObjectSize=True)
    arr = sitk.GetArrayFromImage(relabeled)
    component_count = int(arr.max())
    if component_count == 0:
        return 0, 0.0
    counts = np.bincount(arr.ravel())
    largest = int(counts[1:].max()) if counts.size > 1 else 0
    total = int(mask.sum())
    return component_count, largest / max(total, 1)


def evaluate_mask(
    mask_path: str | Path,
    min_volume_ml: float = 0.01,
    max_volume_ml: float = 250.0,
    min_largest_component_fraction: float = 0.20,
    max_slice_fraction: float = 0.85,
) -> MaskQCResult:
    _require_sitk()
    image = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(image)
    foreground = arr > 0
    foreground_voxels = int(foreground.sum())
    spacing = image.GetSpacing()
    voxel_volume_ml = float(spacing[0] * spacing[1] * spacing[2]) / 1000.0
    volume_ml = foreground_voxels * voxel_volume_ml
    foreground_slices = int(foreground.any(axis=(1, 2)).sum()) if foreground.ndim == 3 else 0
    slice_fraction = foreground_slices / max(int(foreground.shape[0]), 1) if foreground.ndim == 3 else 0.0
    component_count, largest_fraction = _component_stats(foreground)

    reasons: list[str] = []
    if foreground_voxels == 0:
        reasons.append("empty_mask")
    if volume_ml < min_volume_ml:
        reasons.append("volume_too_small")
    if volume_ml > max_volume_ml:
        reasons.append("volume_too_large")
    if component_count > 0 and largest_fraction < min_largest_component_fraction:
        reasons.append("too_fragmented")
    if slice_fraction > max_slice_fraction:
        reasons.append("too_many_slices")

    status = "pass" if not reasons else "fail"
    return MaskQCResult(
        status=status,
        reason=";".join(reasons) if reasons else "ok",
        volume_ml=volume_ml,
        foreground_voxels=foreground_voxels,
        foreground_slices=foreground_slices,
        slice_fraction=slice_fraction,
        component_count=component_count,
        largest_component_fraction=largest_fraction,
    )


def clean_mask_components(
    mask_path: str | Path,
    output_path: str | Path,
    min_component_ml: float = 0.02,
    max_components_per_label: int = 3,
) -> Path:
    _require_sitk()
    image = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(image).astype(np.uint8)
    spacing = image.GetSpacing()
    cleaned = clean_label_array_components(
        arr,
        spacing=spacing,
        min_component_ml=min_component_ml,
        max_components_per_label=max_components_per_label,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned_image = sitk.GetImageFromArray(cleaned)
    cleaned_image.CopyInformation(image)
    sitk.WriteImage(cleaned_image, str(output))
    return output


def clean_label_array_components(
    label_array: np.ndarray,
    spacing: tuple[float, float, float],
    min_component_ml: float = 0.02,
    max_components_per_label: int = 3,
) -> np.ndarray:
    _require_sitk()
    arr = label_array.astype(np.uint8, copy=False)
    voxel_volume_ml = float(spacing[0] * spacing[1] * spacing[2]) / 1000.0
    min_component_voxels = max(1, int(round(min_component_ml / max(voxel_volume_ml, 1e-9))))
    cleaned = np.zeros_like(arr, dtype=np.uint8)

    for label in sorted(int(v) for v in np.unique(arr) if v > 0):
        binary = sitk.GetImageFromArray((arr == label).astype(np.uint8))
        components = sitk.ConnectedComponent(binary)
        relabeled = sitk.RelabelComponent(components, minimumObjectSize=min_component_voxels, sortByObjectSize=True)
        component_arr = sitk.GetArrayFromImage(relabeled)
        keep = (component_arr > 0) & (component_arr <= max_components_per_label)
        cleaned[keep] = label
    return cleaned
