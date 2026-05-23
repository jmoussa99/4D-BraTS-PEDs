#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import LongitudinalMRIDataset, read_manifest
from longitumor.evaluation import dice_score, sensitivity_precision, volume_similarity
from longitumor.inference import load_segmentation_model
from longitumor.qc import clean_label_array_components
from longitumor.utils import parse_patch_size


CLASS_NAMES = ("label_1", "label_2", "label_3", "label_4")


def _normalize_image(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _best_slice(pseudo: np.ndarray, pred: np.ndarray) -> int:
    pseudo_area = pseudo.sum(axis=(1, 2))
    if pseudo_area.max() > 0:
        return int(pseudo_area.argmax())
    pred_area = pred.sum(axis=(1, 2))
    if pred_area.max() > 0:
        return int(pred_area.argmax())
    return int(target.shape[0] // 2)


def _write_overlay(
    image: np.ndarray,
    pseudo: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z = _best_slice(pseudo, pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(_normalize_image(image[z]), cmap="gray")
    if pseudo[z].any():
        ax.contour(pseudo[z], levels=[0.5], colors=["lime"], linewidths=1.2)
    if pred[z].any():
        ax.contour(pred[z], levels=[0.5], colors=["red"], linewidths=1.2)
    ax.set_title(f"{title} z={z}  pseudo=green model=red", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_metric_plot(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['patient_id']} t{row['time_index']}" for row in rows if row["class"] == "mean"]
    dice = [float(row["dice"]) for row in rows if row["class"] == "mean"]
    sensitivity = [float(row["sensitivity"]) for row in rows if row["class"] == "mean"]
    precision = [float(row["precision"]) for row in rows if row["class"] == "mean"]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5))
    ax.plot(x, dice, marker="o", label="Dice")
    ax.plot(x, sensitivity, marker="o", label="Sensitivity")
    ax.plot(x, precision, marker="o", label="Precision")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("pseudo-label agreement")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_volume_plot(rows: list[dict[str, str]], output_path: Path) -> None:
    mean_rows = [row for row in rows if row["class"] == "mean"]
    if not mean_rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['patient_id']} t{row['time_index']}" for row in mean_rows]
    pred = [float(row["pred_volume_ml"]) for row in mean_rows]
    pseudo = [float(row["pseudo_volume_ml"]) for row in mean_rows]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5))
    ax.plot(x, pred, marker="o", label="Model candidate")
    ax.plot(x, pseudo, marker="o", label="Pseudo nnU-Net")
    ax.set_ylabel("volume (mL)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_temporal_consistency(rows: list[dict[str, str]], output_path: Path) -> None:
    mean_rows = [row for row in rows if row["class"] == "mean"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in mean_rows:
        grouped.setdefault((row["patient_id"], row["sequence_index"]), []).append(row)

    temporal_rows: list[dict[str, str]] = []
    for (patient_id, sequence_index), patient_rows in grouped.items():
        ordered = sorted(patient_rows, key=lambda r: int(r["time_index"]))
        for previous, current in zip(ordered, ordered[1:]):
            prev_pred = float(previous["pred_volume_ml"])
            curr_pred = float(current["pred_volume_ml"])
            prev_pseudo = float(previous["pseudo_volume_ml"])
            curr_pseudo = float(current["pseudo_volume_ml"])
            temporal_rows.append(
                {
                    "patient_id": patient_id,
                    "sequence_index": sequence_index,
                    "from_time_index": previous["time_index"],
                    "to_time_index": current["time_index"],
                    "from_delta_t": previous["delta_t"],
                    "to_delta_t": current["delta_t"],
                    "pred_volume_change_ml": f"{curr_pred - prev_pred:.6f}",
                    "pred_relative_change": f"{(curr_pred - prev_pred) / max(prev_pred, 1e-6):.6f}",
                    "pseudo_volume_change_ml": f"{curr_pseudo - prev_pseudo:.6f}",
                    "pseudo_relative_change": f"{(curr_pseudo - prev_pseudo) / max(prev_pseudo, 1e-6):.6f}",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "sequence_index",
                "from_time_index",
                "to_time_index",
                "from_delta_t",
                "to_delta_t",
                "pred_volume_change_ml",
                "pred_relative_change",
                "pseudo_volume_change_ml",
                "pseudo_relative_change",
            ],
        )
        writer.writeheader()
        writer.writerows(temporal_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save longitudinal candidate overlays and pseudo-label agreement plots.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/longitumor_eval"))
    parser.add_argument("--patch-size", default="96,160,160")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of patient sequences to evaluate.")
    parser.add_argument("--postprocess", action="store_true", help="Clean tiny disconnected components before plots/volumes.")
    parser.add_argument("--min-component-ml", type=float, default=0.02)
    parser.add_argument("--max-components-per-label", type=int, default=3)
    args = parser.parse_args()

    model, device = load_segmentation_model(args.checkpoint, args.device)
    records = read_manifest(args.manifest)
    dataset = LongitudinalMRIDataset(records, patch_size=parse_patch_size(args.patch_size))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    rows: list[dict[str, str]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = args.output_dir / "overlays"
    if overlays_dir.exists():
        shutil.rmtree(overlays_dir)
    with torch.no_grad():
        for sequence_index, batch in enumerate(tqdm(loader, desc="evaluating")):
            if args.limit is not None and sequence_index >= args.limit:
                break
            patient_id = str(batch["patient_id"][0])
            x = batch["image"].to(device)
            target = batch["target"].to(device)
            availability = batch["availability"].to(device)
            delta_t = batch["delta_t"].to(device)
            output = model(x, availability=availability, delta_t=delta_t)
            probabilities = output.probabilities.detach().cpu()
            target_cpu = target.detach().cpu()
            image_cpu = batch["image"].detach().cpu()
            spacing_cpu = batch["spacing"].detach().cpu()
            delta_t_cpu = batch["delta_t"].detach().cpu()
            pred_cpu = (probabilities >= args.threshold).float()
            if args.postprocess:
                cleaned = torch.zeros_like(pred_cpu)
                for time_index in range(probabilities.shape[1]):
                    spacing = tuple(float(v) for v in spacing_cpu[0, time_index].tolist())
                    label = torch.argmax(pred_cpu[0, time_index], dim=0).numpy().astype(np.uint8) + 1
                    label[pred_cpu[0, time_index].amax(dim=0).numpy() <= 0] = 0
                    cleaned_label = clean_label_array_components(
                        label,
                        spacing=spacing,
                        min_component_ml=args.min_component_ml,
                        max_components_per_label=args.max_components_per_label,
                    )
                    for class_index in range(len(CLASS_NAMES)):
                        cleaned[0, time_index, class_index] = torch.from_numpy((cleaned_label == class_index + 1).astype(np.float32))
                pred_cpu = cleaned

            metric_pred = pred_cpu if args.postprocess else probabilities
            dice = dice_score(metric_pred.reshape(-1, *metric_pred.shape[2:]), target_cpu.reshape(-1, *target_cpu.shape[2:]), args.threshold)
            sensitivity, precision = sensitivity_precision(
                metric_pred.reshape(-1, *metric_pred.shape[2:]),
                target_cpu.reshape(-1, *target_cpu.shape[2:]),
                args.threshold,
            )
            vol_sim = volume_similarity(
                metric_pred.reshape(-1, *metric_pred.shape[2:]),
                target_cpu.reshape(-1, *target_cpu.shape[2:]),
                args.threshold,
            )

            timepoints = probabilities.shape[1]
            for time_index in range(timepoints):
                flat_index = time_index
                values = {
                    "dice": dice[flat_index],
                    "sensitivity": sensitivity[flat_index],
                    "precision": precision[flat_index],
                    "volume_similarity": vol_sim[flat_index],
                }
                voxel_volume_ml = float(spacing_cpu[0, time_index].prod().item()) / 1000.0
                pred_volumes = pred_cpu[0, time_index].flatten(1).sum(dim=1) * voxel_volume_ml
                target_volumes = target_cpu[0, time_index].flatten(1).sum(dim=1) * voxel_volume_ml
                for class_index, class_name in enumerate(CLASS_NAMES):
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "sequence_index": str(sequence_index),
                            "time_index": str(time_index),
                            "delta_t": f"{delta_t_cpu[0, time_index].item():.6f}",
                            "class": class_name,
                            **{name: f"{tensor[class_index].item():.6f}" for name, tensor in values.items()},
                            "pred_volume_ml": f"{pred_volumes[class_index].item():.6f}",
                            "pseudo_volume_ml": f"{target_volumes[class_index].item():.6f}",
                        }
                    )
                rows.append(
                    {
                        "patient_id": patient_id,
                        "sequence_index": str(sequence_index),
                        "time_index": str(time_index),
                        "delta_t": f"{delta_t_cpu[0, time_index].item():.6f}",
                        "class": "mean",
                        **{name: f"{tensor.mean().item():.6f}" for name, tensor in values.items()},
                        "pred_volume_ml": f"{pred_volumes.sum().item():.6f}",
                        "pseudo_volume_ml": f"{target_volumes.sum().item():.6f}",
                    }
                )

                available = batch["availability"][0, time_index].bool()
                modality_index = int(torch.where(available)[0][0]) if available.any() else 0
                image = image_cpu[0, time_index, modality_index].numpy()
                target_union = target_cpu[0, time_index].amax(dim=0).numpy()
                pred_union = pred_cpu[0, time_index].amax(dim=0).numpy()
                _write_overlay(
                    image=image,
                    pseudo=target_union,
                    pred=pred_union,
                    output_path=overlays_dir / patient_id / f"time_{time_index:02d}.png",
                    title=f"{patient_id} time {time_index}",
                )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "sequence_index",
                "time_index",
                "delta_t",
                "class",
                "dice",
                "sensitivity",
                "precision",
                "volume_similarity",
                "pred_volume_ml",
                "pseudo_volume_ml",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_metric_plot(rows, args.output_dir / "metric_trends.png")
    _write_volume_plot(rows, args.output_dir / "volume_trends.png")
    _write_temporal_consistency(rows, args.output_dir / "temporal_consistency.csv")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote overlays under {args.output_dir / 'overlays'}")
    print(f"Wrote metric plot to {args.output_dir / 'metric_trends.png'}")
    print(f"Wrote volume plot to {args.output_dir / 'volume_trends.png'}")
    print(f"Wrote temporal consistency to {args.output_dir / 'temporal_consistency.csv'}")


if __name__ == "__main__":
    main()
