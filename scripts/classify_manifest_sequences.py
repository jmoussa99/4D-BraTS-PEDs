#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import MODALITY_NAMES, looks_like_localizer, read_manifest
from longitumor.sequence_classifier import classify_volume_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MRISeqClassifier on the modality files listed in a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mriseqclassifier-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classifier-python", default=sys.executable)
    parser.add_argument("--num-slices", type=int, default=9)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    tasks: list[tuple[str, str, str, Path]] = []
    for record in read_manifest(args.manifest):
        for expected, path in zip(MODALITY_NAMES, record.modalities):
            if not path:
                continue
            source = Path(path)
            if looks_like_localizer(source):
                continue
            tasks.append((record.patient_id, record.visit_id, expected, source))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    for patient_id, visit_id, expected, path in tasks:
        try:
            prediction = classify_volume_sequence(
                path,
                classifier_repo=args.mriseqclassifier_repo,
                python_executable=args.classifier_python,
                num_slices=args.num_slices,
            )
            predicted = prediction.modality or ""
            status = "ok"
            if expected == "t1c":
                match = predicted == "t1"
            else:
                match = predicted == expected
            error = ""
            print(f"{patient_id} {visit_id} {expected}\t{predicted}\t{prediction.confidence:.4f}\t{prediction.votes}")
        except Exception as exc:
            prediction = None
            predicted = ""
            status = "error"
            match = False
            error = repr(exc)
            print(f"{patient_id} {visit_id} {expected}\tERROR\t{exc}")

        rows.append(
            {
                "patient_id": patient_id,
                "visit_id": visit_id,
                "expected_manifest_modality": expected,
                "classifier_modality": predicted,
                "classifier_label": prediction.label if prediction else "",
                "confidence": f"{prediction.confidence:.4f}" if prediction else "",
                "votes": repr(prediction.votes) if prediction else "",
                "match": str(match),
                "status": status,
                "error": error,
                "path": str(path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "visit_id",
                "expected_manifest_modality",
                "classifier_modality",
                "classifier_label",
                "confidence",
                "votes",
                "match",
                "status",
                "error",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} manifest sequence-QC rows to {args.output}")


if __name__ == "__main__":
    main()
