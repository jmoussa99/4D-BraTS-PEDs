#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import VisitRecord, discover_cases, write_manifest
from longitumor.sequence_classifier import classify_volume_sequence


def _attach_generated_masks(records: list[VisitRecord], mask_paths: list[Path]) -> list[VisitRecord]:
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
    parser = argparse.ArgumentParser(description="Create an OmniMamba4DMRI manifest from case folders.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("longitumor_manifest.csv"))
    parser.add_argument(
        "--mriseqclassifier-repo",
        type=Path,
        default=None,
        help="Optional local clone of https://github.com/JinqianPan/MRISeqClassifier with best models installed.",
    )
    parser.add_argument("--classifier-threshold", type=float, default=0.70)
    parser.add_argument("--classifier-python", default=sys.executable)
    parser.add_argument(
        "--generate-masks-checkpoint",
        type=Path,
        default=None,
        help="Optional trained checkpoint used to generate predicted masks for discovered visits.",
    )
    parser.add_argument(
        "--generated-mask-dir",
        type=Path,
        default=Path("predicted_masks"),
        help="Output directory for generated masks when --generate-masks-checkpoint is used.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--pediatric-model-repo",
        type=Path,
        default=None,
        help="Local clone of NUBagciLab/Pediatric-Brain-Tumor-Segmentation-Model used to generate masks.",
    )
    parser.add_argument("--pediatric-channel-order", default="flair,t1,t1c,t2")
    parser.add_argument("--pediatric-folds", default="0,1,2,3,4")
    parser.add_argument("--pediatric-python", default=sys.executable)
    parser.add_argument("--pediatric-command", default="nnUNetv2_predict")
    parser.add_argument(
        "--include-visit-token",
        action="append",
        default=[],
        help="Keep only visits whose visit_id contains this token. Can be repeated, for example --include-visit-token brain.",
    )
    parser.add_argument(
        "--exclude-visit-token",
        action="append",
        default=[],
        help="Drop visits whose visit_id contains this token. Can be repeated, for example --exclude-visit-token spine.",
    )
    args = parser.parse_args()
    if args.generate_masks_checkpoint is not None and args.pediatric_model_repo is not None:
        parser.error("Use only one mask generator: --generate-masks-checkpoint or --pediatric-model-repo.")

    classifier = None
    if args.mriseqclassifier_repo is not None:
        def classifier(path: Path) -> tuple[str, float]:
            prediction = classify_volume_sequence(
                path,
                classifier_repo=args.mriseqclassifier_repo,
                python_executable=args.classifier_python,
            )
            return prediction.modality or "", prediction.confidence

    records = discover_cases(args.data_dir, modality_classifier=classifier, classifier_threshold=args.classifier_threshold)
    include_tokens = [token.lower() for token in args.include_visit_token]
    exclude_tokens = [token.lower() for token in args.exclude_visit_token]
    if include_tokens:
        records = [
            record
            for record in records
            if any(token in record.visit_id.lower() for token in include_tokens)
        ]
    if exclude_tokens:
        records = [
            record
            for record in records
            if not any(token in record.visit_id.lower() for token in exclude_tokens)
        ]
    if args.generate_masks_checkpoint is not None:
        from longitumor.inference import load_segmentation_model, predict_visit_mask

        model, device = load_segmentation_model(args.generate_masks_checkpoint, args.device)
        mask_paths: list[Path] = []
        for record in tqdm(records, desc="generating masks"):
            output = args.generated_mask_dir / record.patient_id / f"{record.visit_id}_seg.nii.gz"
            mask_paths.append(predict_visit_mask(model, record, output, device, threshold=args.mask_threshold))
        records = _attach_generated_masks(records, mask_paths)
    elif args.pediatric_model_repo is not None:
        from longitumor.inference import generate_pediatric_brain_tumor_masks

        channel_order = tuple(part.strip() for part in args.pediatric_channel_order.split(",") if part.strip())
        folds = tuple(part.strip() for part in args.pediatric_folds.split(",") if part.strip())
        mask_paths = generate_pediatric_brain_tumor_masks(
            records,
            model_repo=args.pediatric_model_repo,
            output_dir=args.generated_mask_dir,
            channel_order=channel_order,
            folds=folds,
            device=args.device,
            python_executable=args.pediatric_python,
            command=args.pediatric_command,
        )
        records = _attach_generated_masks(records, mask_paths)

    write_manifest(records, args.output)
    labeled = sum(1 for record in records if record.mask_path)
    print(f"Wrote {len(records)} records ({labeled} labeled) to {args.output}")


if __name__ == "__main__":
    main()
