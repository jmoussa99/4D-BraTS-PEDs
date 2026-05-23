#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import MODALITY_NAMES


def _copy_if_present(source: str | Path | None, destination: Path) -> bool:
    if not source:
        return False
    src = Path(source)
    if not src.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MRI images and matching candidate masks into review folders.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pred-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/review_cases"))
    parser.add_argument("--copy-original-reference", action="store_true", help="Deprecated alias for --copy-original-pseudo-mask.")
    parser.add_argument("--copy-original-pseudo-mask", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with args.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            patient_id = row["patient_id"]
            visit_id = row["visit_id"]
            case_dir = args.output_dir / patient_id / visit_id
            images_dir = case_dir / "images"
            masks_dir = case_dir / "masks"
            pred_source_dir = args.pred_mask_dir / patient_id
            pred_stem = f"{visit_id}_seg"

            copied_images: list[str] = []
            copied_masks: list[str] = []
            for modality in MODALITY_NAMES:
                image_out = images_dir / f"{modality}.nii.gz"
                if _copy_if_present(row.get(modality), image_out):
                    copied_images.append(str(image_out))

                pred = pred_source_dir / f"{pred_stem}_{modality}.nii.gz"
                pseudo = pred_source_dir / f"{pred_stem}_pseudo_nnunet_{modality}.nii.gz"
                legacy_pseudo = pred_source_dir / f"{pred_stem}_reference_{modality}.nii.gz"
                pred_out = masks_dir / f"model_candidate_{modality}.nii.gz"
                pseudo_out = masks_dir / f"pseudo_nnunet_{modality}.nii.gz"
                if _copy_if_present(pred, pred_out):
                    copied_masks.append(str(pred_out))
                if _copy_if_present(pseudo, pseudo_out) or _copy_if_present(legacy_pseudo, pseudo_out):
                    copied_masks.append(str(pseudo_out))

            canonical_pred = pred_source_dir / f"{pred_stem}.nii.gz"
            canonical_pseudo = pred_source_dir / f"{pred_stem}_pseudo_nnunet.nii.gz"
            legacy_canonical_pseudo = pred_source_dir / f"{pred_stem}_reference.nii.gz"
            if _copy_if_present(canonical_pred, masks_dir / "model_candidate_common_grid.nii.gz"):
                copied_masks.append(str(masks_dir / "model_candidate_common_grid.nii.gz"))
            if _copy_if_present(canonical_pseudo, masks_dir / "pseudo_nnunet_common_grid.nii.gz") or _copy_if_present(
                legacy_canonical_pseudo, masks_dir / "pseudo_nnunet_common_grid.nii.gz"
            ):
                copied_masks.append(str(masks_dir / "pseudo_nnunet_common_grid.nii.gz"))
            if (args.copy_original_pseudo_mask or args.copy_original_reference) and _copy_if_present(
                row.get("mask"), masks_dir / "original_manifest_pseudo_nnunet.nii.gz"
            ):
                copied_masks.append(str(masks_dir / "original_manifest_pseudo_nnunet.nii.gz"))

            rows.append(
                {
                    "patient_id": patient_id,
                    "visit_id": visit_id,
                    "review_folder": str(case_dir),
                    "images_copied": str(len(copied_images)),
                    "masks_copied": str(len(copied_masks)),
                }
            )

    index_path = args.output_dir / "review_index.csv"
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "visit_id", "review_folder", "images_copied", "masks_copied"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} review cases to {args.output_dir}")
    print(f"Wrote review index to {index_path}")


if __name__ == "__main__":
    main()
