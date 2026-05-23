#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import SEG_EXTENSIONS, looks_like_localizer
from longitumor.sequence_classifier import classify_volume_sequence


def _is_image(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and path.name.lower().endswith(SEG_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MRI sequence classification for T1/T1c/T2/FLAIR QC.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder scanned recursively.")
    parser.add_argument(
        "--mriseqclassifier-repo",
        type=Path,
        required=True,
        help="Local MRISeqClassifier checkout containing 05_toolkit.py and 02_models/best_model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classifier-python", default=sys.executable)
    parser.add_argument("--num-slices", type=int, default=9)
    parser.add_argument("--include-path-token", action="append", default=[])
    parser.add_argument("--exclude-path-token", action="append", default=[])
    parser.add_argument(
        "--include-localizers",
        action="store_true",
        help="Classify scout/localizer series too. By default these are skipped.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    include_tokens = [token.lower() for token in args.include_path_token]
    exclude_tokens = [token.lower() for token in args.exclude_path_token]
    paths = [args.input] if args.input.is_file() else sorted(path for path in args.input.rglob("*") if _is_image(path))
    if include_tokens:
        paths = [path for path in paths if any(token in str(path).lower() for token in include_tokens)]
    if exclude_tokens:
        paths = [path for path in paths if not any(token in str(path).lower() for token in exclude_tokens)]
    if not args.include_localizers:
        paths = [path for path in paths if not looks_like_localizer(path)]
    if args.limit is not None:
        paths = paths[: args.limit]

    rows = []
    for path in paths:
        try:
            prediction = classify_volume_sequence(
                path,
                classifier_repo=args.mriseqclassifier_repo,
                python_executable=args.classifier_python,
                num_slices=args.num_slices,
            )
            row = {
                "path": str(path),
                "label": prediction.label,
                "modality": prediction.modality or "",
                "confidence": f"{prediction.confidence:.4f}",
                "votes": repr(prediction.votes),
                "status": "ok",
                "error": "",
            }
            print(f"{path}\t{prediction.modality}\t{prediction.confidence:.4f}\t{prediction.votes}")
        except Exception as exc:
            row = {
                "path": str(path),
                "label": "",
                "modality": "",
                "confidence": "",
                "votes": "",
                "status": "error",
                "error": repr(exc),
            }
            print(f"{path}\tERROR\t{exc}")
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "label", "modality", "confidence", "votes", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sequence-classification rows to {args.output}")


if __name__ == "__main__":
    main()
