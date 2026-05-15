#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import FutureSegmentationDataset, _load_visit_arrays, _pad_spatial_to_shape, random_patch_slices, read_manifest
from longitumor.evaluation import dice_score, sensitivity_precision, volume_similarity
from longitumor.inference import load_segmentation_model, probabilities_to_labelmap
from longitumor.utils import parse_patch_size

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover
    sitk = None
    _sitk_error = exc
else:
    _sitk_error = None


def _parse_optional_patch_size(value: str) -> tuple[int, int, int] | None:
    if value.lower() in {"none", "full"}:
        return None
    return parse_patch_size(value)


def _normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _best_slice(target: np.ndarray, pred: np.ndarray) -> int:
    area = target.sum(axis=(1, 2))
    if area.max() > 0:
        return int(area.argmax())
    pred_area = pred.sum(axis=(1, 2))
    if pred_area.max() > 0:
        return int(pred_area.argmax())
    return int(target.shape[0] // 2)


def _write_overlay(image: np.ndarray, target: np.ndarray | None, pred: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z = _best_slice(target if target is not None else np.zeros_like(pred), pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(_normalize(image[z]), cmap="gray")
    if target is not None and target[z].any():
        ax.contour(target[z], levels=[0.5], colors=["lime"], linewidths=1.2)
    if pred[z].any():
        ax.contour(pred[z], levels=[0.5], colors=["red"], linewidths=1.2)
    ax.set_title(f"{title} z={z} target=green forecast=red", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_nifti(label: np.ndarray, reference_path: str, output_path: Path) -> None:
    if sitk is None:
        raise ImportError("SimpleITK is required to write NIfTI files") from _sitk_error
    reference = sitk.ReadImage(str(reference_path))
    image = sitk.GetImageFromArray(label)
    if image.GetSize() == reference.GetSize():
        image.CopyInformation(reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next-visit future tumor segmentations.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/longitumor_future_predictions"))
    parser.add_argument("--input-timepoints", type=int, default=3)
    parser.add_argument("--patch-size", default="96,160,160", help="Use 'none' or 'full' for full-volume inference.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Forecast one unseen next mask after each patient's latest visit instead of evaluating known next visits.",
    )
    args = parser.parse_args()

    model, device = load_segmentation_model(args.checkpoint, args.device)
    records = read_manifest(args.manifest)
    patch_size = _parse_optional_patch_size(args.patch_size)
    if args.latest:
        if sitk is None:
            raise ImportError("SimpleITK is required for --latest forecasting") from _sitk_error
        grouped: dict[str, list] = {}
        for record in records:
            if record.mask_path:
                grouped.setdefault(record.patient_id, []).append(record)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for patient_id, visits in grouped.items():
                ordered = sorted(visits, key=lambda r: (r.delta_t, r.visit_id))
                if len(ordered) < args.input_timepoints:
                    continue
                input_visits = ordered[-args.input_timepoints :]
                reference_path = input_visits[-1].mask_path or next(path for path in input_visits[-1].modalities if path)
                reference = sitk.ReadImage(str(reference_path))
                images = []
                availability = []
                for visit in input_visits:
                    image, _, available, _ = _load_visit_arrays(visit, reference=reference)
                    images.append(image)
                    availability.append(torch.tensor(available, dtype=torch.float32))
                image_sequence = np.stack(images, axis=0)
                if patch_size is not None:
                    zsl, ysl, xsl = random_patch_slices(image_sequence.shape[-3:], patch_size)
                    image_sequence = image_sequence[:, :, zsl, ysl, xsl]
                    image_sequence = _pad_spatial_to_shape(image_sequence, patch_size)
                x = torch.from_numpy(image_sequence).unsqueeze(0).to(device)
                available_tensor = torch.stack(availability, dim=0).unsqueeze(0).to(device)
                delta_t = torch.tensor([[visit.delta_t for visit in input_visits]], dtype=torch.float32, device=device)
                output = model(x, availability=available_tensor, delta_t=delta_t)
                probabilities = output.probabilities[0, -1].detach().cpu()
                pred = (probabilities >= args.threshold).float().amax(dim=0).numpy()
                stem = f"forecast_after_{input_visits[-1].visit_id}"
                patient_dir = args.output_dir / patient_id
                _write_overlay(image_sequence[-1, 0], None, pred, patient_dir / f"{stem}.png", stem)
                if patch_size is None:
                    label = probabilities_to_labelmap(probabilities, threshold=args.threshold)
                    _write_nifti(label, str(reference_path), patient_dir / f"{stem}.nii.gz")
        print(f"Wrote latest-visit forecasts under {args.output_dir}")
        if patch_size is not None:
            print("Patch mode wrote PNG forecasts. Use --patch-size full to also write full-volume NIfTI masks.")
        return

    dataset = FutureSegmentationDataset(records, input_timepoints=args.input_timepoints, patch_size=patch_size)
    if len(dataset) == 0:
        raise ValueError("No future-prediction windows found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            if args.limit is not None and index >= args.limit:
                break
            sample = dataset[index]
            input_visits, future_visit = dataset.windows[index]
            x = sample["image"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            availability = sample["availability"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            delta_t = sample["delta_t"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            output = model(x, availability=availability, delta_t=delta_t)
            probabilities = output.probabilities[0, -1].detach().cpu()
            target = sample["target"]  # type: ignore[assignment]
            pred = (probabilities >= args.threshold).float()

            dice = dice_score(probabilities.unsqueeze(0), target.unsqueeze(0), args.threshold)[0]  # type: ignore[union-attr]
            sensitivity, precision = sensitivity_precision(probabilities.unsqueeze(0), target.unsqueeze(0), args.threshold)  # type: ignore[union-attr]
            vol_sim = volume_similarity(probabilities.unsqueeze(0), target.unsqueeze(0), args.threshold)[0]  # type: ignore[union-attr]
            voxel_volume_ml = float(sample["spacing"].prod().item()) / 1000.0  # type: ignore[union-attr]
            pred_volume_ml = float(pred.flatten(1).sum().item()) * voxel_volume_ml
            target_volume_ml = float(target.flatten(1).sum().item()) * voxel_volume_ml  # type: ignore[union-attr]
            rows.append(
                {
                    "patient_id": str(sample["patient_id"]),
                    "input_visit_ids": str(sample["input_visit_ids"]),
                    "future_visit_id": str(sample["future_visit_id"]),
                    "dice": f"{dice.mean().item():.6f}",
                    "sensitivity": f"{sensitivity[0].mean().item():.6f}",
                    "precision": f"{precision[0].mean().item():.6f}",
                    "volume_similarity": f"{vol_sim.mean().item():.6f}",
                    "pred_volume_ml": f"{pred_volume_ml:.6f}",
                    "target_volume_ml": f"{target_volume_ml:.6f}",
                }
            )

            image = sample["image"][-1, 0].numpy()  # type: ignore[index,union-attr]
            target_union = target.amax(dim=0).numpy()  # type: ignore[union-attr]
            pred_union = pred.amax(dim=0).numpy()
            patient_dir = args.output_dir / str(sample["patient_id"])
            stem = f"future_{index:03d}_{future_visit.visit_id}"
            _write_overlay(image, target_union, pred_union, patient_dir / f"{stem}.png", stem)

            if patch_size is None and future_visit.mask_path:
                label = probabilities_to_labelmap(probabilities, threshold=args.threshold)
                _write_nifti(label, future_visit.mask_path, patient_dir / f"{stem}.nii.gz")

    metrics_path = args.output_dir / "future_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "input_visit_ids",
                "future_visit_id",
                "dice",
                "sensitivity",
                "precision",
                "volume_similarity",
                "pred_volume_ml",
                "target_volume_ml",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote future metrics to {metrics_path}")
    print(f"Wrote future overlays under {args.output_dir}")
    if patch_size is None:
        print("Full-volume mode also wrote NIfTI forecast masks.")


if __name__ == "__main__":
    main()
