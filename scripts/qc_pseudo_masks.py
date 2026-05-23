#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import VisitRecord, read_manifest, write_manifest
from longitumor.qc import clean_mask_components, evaluate_mask, modality_qc_reasons


def _attach_previous_masks(records: list[VisitRecord]) -> list[VisitRecord]:
    previous_by_patient: dict[str, str] = {}
    updated: list[VisitRecord] = []
    for record in sorted(records, key=lambda r: (r.patient_id, r.delta_t, r.visit_id)):
        updated.append(replace(record, previous_mask_path=previous_by_patient.get(record.patient_id)))
        if record.mask_path:
            previous_by_patient[record.patient_id] = record.mask_path
    return updated


def _filter_min_timepoints(records: list[VisitRecord], min_timepoints: int) -> list[VisitRecord]:
    grouped: dict[str, list[VisitRecord]] = {}
    for record in records:
        grouped.setdefault(record.patient_id, []).append(record)
    kept: list[VisitRecord] = []
    for visits in grouped.values():
        if len(visits) >= min_timepoints:
            kept.extend(visits)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QC pseudo masks, clean disconnected islands, and write a retraining manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-mask-dir", type=Path, default=Path("runs/pseudo_masks_qc"))
    parser.add_argument("--qc-report", type=Path, default=Path("runs/pseudo_mask_qc.csv"))
    parser.add_argument("--min-modalities", type=int, default=2)
    parser.add_argument("--min-slices", type=int, default=8)
    parser.add_argument("--min-inplane", type=int, default=64)
    parser.add_argument("--min-volume-ml", type=float, default=0.01)
    parser.add_argument("--max-volume-ml", type=float, default=250.0)
    parser.add_argument("--min-largest-component-fraction", type=float, default=0.20)
    parser.add_argument("--max-slice-fraction", type=float, default=0.85)
    parser.add_argument("--min-component-ml", type=float, default=0.02)
    parser.add_argument("--max-components-per-label", type=int, default=3)
    parser.add_argument("--min-timepoints", type=int, default=2)
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Write the QC report but keep failed records in the output manifest.",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    kept: list[VisitRecord] = []
    for record in read_manifest(args.manifest):
        reasons = modality_qc_reasons(
            record,
            min_modalities=args.min_modalities,
            min_slices=args.min_slices,
            min_inplane=args.min_inplane,
        )
        if not record.mask_path:
            reasons.append("missing_pseudo_mask")
            qc = None
        else:
            qc = evaluate_mask(
                record.mask_path,
                min_volume_ml=args.min_volume_ml,
                max_volume_ml=args.max_volume_ml,
                min_largest_component_fraction=args.min_largest_component_fraction,
                max_slice_fraction=args.max_slice_fraction,
            )
            if qc.status != "pass":
                reasons.append(qc.reason)

        status = "pass" if not reasons else "fail"
        cleaned_mask = ""
        if qc is not None and (status == "pass" or args.keep_failed):
            output = args.output_mask_dir / record.patient_id / f"{record.visit_id}_pseudo_clean.nii.gz"
            cleaned_mask = str(
                clean_mask_components(
                    record.mask_path,
                    output,
                    min_component_ml=args.min_component_ml,
                    max_components_per_label=args.max_components_per_label,
                )
            )
            if status == "pass" or args.keep_failed:
                kept.append(replace(record, mask_path=cleaned_mask))

        rows.append(
            {
                "patient_id": record.patient_id,
                "visit_id": record.visit_id,
                "status": status,
                "reason": ";".join(reasons) if reasons else "ok",
                "mask_path": record.mask_path or "",
                "cleaned_mask_path": cleaned_mask,
                "volume_ml": f"{qc.volume_ml:.6f}" if qc else "",
                "foreground_voxels": str(qc.foreground_voxels) if qc else "",
                "foreground_slices": str(qc.foreground_slices) if qc else "",
                "slice_fraction": f"{qc.slice_fraction:.6f}" if qc else "",
                "component_count": str(qc.component_count) if qc else "",
                "largest_component_fraction": f"{qc.largest_component_fraction:.6f}" if qc else "",
            }
        )

    kept = _filter_min_timepoints(kept, args.min_timepoints)
    kept = _attach_previous_masks(kept)
    write_manifest(kept, args.output_manifest)

    args.qc_report.parent.mkdir(parents=True, exist_ok=True)
    with args.qc_report.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "visit_id",
                "status",
                "reason",
                "mask_path",
                "cleaned_mask_path",
                "volume_ml",
                "foreground_voxels",
                "foreground_slices",
                "slice_fraction",
                "component_count",
                "largest_component_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for row in rows if row["status"] == "pass")
    print(f"QC passed {passed}/{len(rows)} visits")
    print(f"Wrote cleaned manifest with {len(kept)} visits to {args.output_manifest}")
    print(f"Wrote QC report to {args.qc_report}")


if __name__ == "__main__":
    main()
