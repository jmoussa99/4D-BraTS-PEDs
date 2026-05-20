#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import VisitRecord, read_manifest, write_manifest


def _pick_indices(count: int, timepoints: int) -> list[int]:
    if timepoints <= 0:
        raise ValueError("timepoints must be positive")
    if count <= timepoints:
        return list(range(count))
    if timepoints == 1:
        return [0]
    raw = [round(i * (count - 1) / (timepoints - 1)) for i in range(timepoints)]
    indices: list[int] = []
    for index in raw:
        if index not in indices:
            indices.append(index)
    cursor = 0
    while len(indices) < timepoints:
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    return sorted(indices[:timepoints])


def _attach_previous_masks(records: list[VisitRecord]) -> list[VisitRecord]:
    previous_by_patient: dict[str, str] = {}
    updated: list[VisitRecord] = []
    for record in records:
        updated.append(replace(record, previous_mask_path=previous_by_patient.get(record.patient_id)))
        if record.mask_path:
            previous_by_patient[record.patient_id] = record.mask_path
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Select baseline/mid/end longitudinal visits per patient.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timepoints", type=int, default=3, help="Usually 3: baseline, mid-treatment, end follow-up.")
    parser.add_argument("--min-timepoints", type=int, default=2)
    parser.add_argument("--require-mask", action="store_true")
    args = parser.parse_args()

    grouped: dict[str, list[VisitRecord]] = {}
    for record in read_manifest(args.manifest):
        if args.require_mask and not record.mask_path:
            continue
        grouped.setdefault(record.patient_id, []).append(record)

    selected: list[VisitRecord] = []
    for patient_id, visits in grouped.items():
        ordered = sorted(visits, key=lambda r: (r.delta_t, r.visit_id))
        if len(ordered) < args.min_timepoints:
            continue
        selected.extend(ordered[index] for index in _pick_indices(len(ordered), args.timepoints))

    selected = sorted(selected, key=lambda r: (r.patient_id, r.delta_t, r.visit_id))
    selected = _attach_previous_masks(selected)
    write_manifest(selected, args.output)
    print(f"Wrote {len(selected)} visits from {len(grouped)} patients to {args.output}")


if __name__ == "__main__":
    main()
