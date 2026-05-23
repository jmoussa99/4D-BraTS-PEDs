#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import VisitRecord, read_manifest, write_manifest
from longitumor.inference import load_segmentation_model, predict_visit_mask, write_pseudo_mask_copies


def _with_generated_masks(records: list[VisitRecord], mask_paths: list[Path]) -> list[VisitRecord]:
    previous_by_patient: dict[str, str] = {}
    updated: list[VisitRecord] = []
    for record, mask_path in zip(records, mask_paths):
        mask = str(mask_path)
        updated.append(
            replace(
                record,
                mask_path=mask,
                previous_mask_path=previous_by_patient.get(record.patient_id),
            )
        )
        previous_by_patient[record.patient_id] = mask
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predicted segmentation masks for manifest visits.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Trained LongiTumorMamba checkpoint.")
    parser.add_argument(
        "--pediatric-model-repo",
        type=Path,
        default=None,
        help="Local clone of NUBagciLab/Pediatric-Brain-Tumor-Segmentation-Model.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("predicted_masks"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of records to process.")
    parser.add_argument("--pediatric-channel-order", default="flair,t1,t1c,t2")
    parser.add_argument("--pediatric-folds", default="0,1,2,3,4")
    parser.add_argument("--pediatric-python", default=sys.executable)
    parser.add_argument("--pediatric-command", default="nnUNetv2_predict")
    parser.add_argument(
        "--write-modality-space-masks",
        action="store_true",
        help="For checkpoint inference, also write nearest-neighbor mask copies aligned to each selected modality grid.",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Remove tiny disconnected components from checkpoint-generated masks before writing review copies.",
    )
    parser.add_argument("--min-component-ml", type=float, default=0.02)
    parser.add_argument("--max-components-per-label", type=int, default=3)
    parser.add_argument(
        "--write-reference-mask-copies",
        action="store_true",
        help="Deprecated alias for --write-pseudo-mask-copies.",
    )
    parser.add_argument(
        "--write-pseudo-mask-copies",
        action="store_true",
        help="Also copy/resample the manifest pseudo mask next to each generated prediction for visual comparison.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Optional CSV manifest with mask/previous_mask columns populated from generated masks.",
    )
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.pediatric_model_repo is None):
        parser.error("Choose exactly one mask generator: --checkpoint or --pediatric-model-repo.")

    records = read_manifest(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]

    if args.checkpoint is not None:
        model, device = load_segmentation_model(args.checkpoint, args.device)
        written = []
        for record in tqdm(records, desc="generating masks"):
            output = args.output_dir / record.patient_id / f"{record.visit_id}_seg.nii.gz"
            written.append(
                predict_visit_mask(
                    model,
                    record,
                    output,
                    device,
                    threshold=args.threshold,
                    write_modality_space_masks=args.write_modality_space_masks,
                    postprocess=args.postprocess,
                    min_component_ml=args.min_component_ml,
                    max_components_per_label=args.max_components_per_label,
                )
            )
            if args.write_pseudo_mask_copies or args.write_reference_mask_copies:
                write_pseudo_mask_copies(
                    record,
                    output,
                    write_modality_space_masks=args.write_modality_space_masks,
                )
    else:
        from longitumor.inference import generate_pediatric_brain_tumor_masks

        channel_order = tuple(part.strip() for part in args.pediatric_channel_order.split(",") if part.strip())
        folds = tuple(part.strip() for part in args.pediatric_folds.split(",") if part.strip())
        written = generate_pediatric_brain_tumor_masks(
            records,
            model_repo=args.pediatric_model_repo,
            output_dir=args.output_dir,
            channel_order=channel_order,
            folds=folds,
            device=args.device,
            python_executable=args.pediatric_python,
            command=args.pediatric_command,
        )
    print(f"Wrote {len(written)} predicted masks to {args.output_dir}")

    if args.output_manifest is not None:
        updated_records = _with_generated_masks(records, written)
        write_manifest(updated_records, args.output_manifest)
        print(f"Wrote generated-mask manifest to {args.output_manifest}")


if __name__ == "__main__":
    main()
