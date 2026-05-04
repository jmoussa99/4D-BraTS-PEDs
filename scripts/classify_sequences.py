#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import SEG_EXTENSIONS
from longitumor.sequence_classifier import SequenceClassifierPredictor


def _is_image(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and path.name.lower().endswith(SEG_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify anonymous MRI volumes as T1/T2/T1c/FLAIR.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder to scan recursively.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    predictor = SequenceClassifierPredictor(args.checkpoint, args.device)
    paths = [args.input] if args.input.is_file() else sorted(path for path in args.input.rglob("*") if _is_image(path))
    rows = []
    for path in paths:
        prediction = predictor.predict(path)
        row = {"path": str(path), "modality": prediction.modality, "confidence": f"{prediction.confidence:.4f}"}
        row.update({f"p_{key}": f"{value:.4f}" for key, value in prediction.probabilities.items()})
        rows.append(row)
        print(f"{path}\t{prediction.modality}\t{prediction.confidence:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["path", "modality", "confidence", "p_t1", "p_t2", "p_t1c", "p_flair"]
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
