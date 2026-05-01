from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred_bin = (pred >= threshold).float()
    target = target.float()
    dims = tuple(range(2, pred.ndim))
    intersection = (pred_bin * target).sum(dim=dims)
    denom = pred_bin.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * intersection + eps) / (denom + eps)


def sensitivity_precision(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_bin = pred >= threshold
    target_bin = target.bool()
    dims = tuple(range(2, pred.ndim))
    tp = (pred_bin & target_bin).sum(dim=dims).float()
    fn = ((~pred_bin) & target_bin).sum(dim=dims).float()
    fp = (pred_bin & (~target_bin)).sum(dim=dims).float()
    return tp / (tp + fn + eps), tp / (tp + fp + eps)


def volume_similarity(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred_vol = (pred >= threshold).float().sum(dim=tuple(range(2, pred.ndim)))
    target_vol = target.float().sum(dim=tuple(range(2, target.ndim)))
    return 1.0 - (pred_vol - target_vol).abs() / (pred_vol + target_vol + eps)


def hausdorff95(pred_mask: np.ndarray, target_mask: np.ndarray, spacing: tuple[float, float, float] = (1, 1, 1)) -> float:
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    if not pred_mask.any() and not target_mask.any():
        return 0.0
    if not pred_mask.any() or not target_mask.any():
        return float("inf")
    pred_surface = pred_mask ^ ndimage.binary_erosion(pred_mask)
    target_surface = target_mask ^ ndimage.binary_erosion(target_mask)
    dt_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing[::-1])
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing[::-1])
    distances = np.concatenate([dt_target[pred_surface], dt_pred[target_surface]])
    return float(np.percentile(distances, 95))


def volume_features(probabilities: torch.Tensor, spacing: torch.Tensor | None = None, threshold: float = 0.5) -> torch.Tensor:
    mask = (probabilities >= threshold).float()
    voxel_counts = mask.flatten(3).sum(dim=-1)
    if spacing is None:
        return voxel_counts
    voxel_volume = spacing.prod(dim=-1)
    while voxel_volume.ndim < voxel_counts.ndim:
        voxel_volume = voxel_volume.unsqueeze(-1)
    return voxel_counts * voxel_volume


def longitudinal_volume_changes(volumes: torch.Tensor, delta_t: torch.Tensor, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    absolute = volumes[:, 1:] - volumes[:, :-1]
    relative = absolute / volumes[:, :-1].clamp_min(eps)
    dt = delta_t[:, 1:].clamp_min(eps).unsqueeze(-1)
    monthly = relative / dt
    return {
        "absolute_volume_change": absolute,
        "relative_volume_change": relative,
        "monthly_growth_rate": monthly,
    }


def centroid(mask: torch.Tensor, spacing: torch.Tensor | None = None) -> torch.Tensor:
    if mask.ndim != 5:
        raise ValueError("Expected mask shaped (B, C, D, H, W)")
    b, c, d, h, w = mask.shape
    coords = torch.stack(torch.meshgrid(
        torch.arange(d, device=mask.device),
        torch.arange(h, device=mask.device),
        torch.arange(w, device=mask.device),
        indexing="ij",
    ), dim=-1).float()
    if spacing is not None:
        coords = coords * spacing[:, None, None, None, [2, 1, 0]]
    weights = mask.float().unsqueeze(-1)
    denom = weights.sum(dim=(2, 3, 4)).clamp_min(1e-6)
    return (weights * coords).sum(dim=(2, 3, 4)) / denom
