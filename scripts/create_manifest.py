#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import discover_cases, write_manifest
from longitumor.sequence_classifier import classify_volume_sequence


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
    args = parser.parse_args()

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
    write_manifest(records, args.output)
    labeled = sum(1 for record in records if record.mask_path)
    print(f"Wrote {len(records)} records ({labeled} labeled) to {args.output}")


if __name__ == "__main__":
    main()
