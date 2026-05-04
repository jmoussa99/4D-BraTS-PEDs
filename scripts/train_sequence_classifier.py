#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.sequence_classifier import (
    labeled_paths_from_manifest,
    read_sequence_training_csv,
    train_sequence_classifier,
    write_sequence_training_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MRI-sequence classifier for T1/T2/T1c/FLAIR preprocessing.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Existing manifest with known t1,t2,t1c,flair columns.")
    source.add_argument("--training-csv", type=Path, help="CSV with columns: path,modality.")
    parser.add_argument("--output", type=Path, default=Path("runs/sequence_classifier.pt"))
    parser.add_argument("--export-training-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    pairs = labeled_paths_from_manifest(args.manifest) if args.manifest else read_sequence_training_csv(args.training_csv)
    if args.export_training_csv:
        write_sequence_training_csv(pairs, args.export_training_csv)
    checkpoint = train_sequence_classifier(
        pairs,
        output_path=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        device_name=args.device,
    )
    print(f"Trained sequence classifier on {len(pairs)} volumes: {checkpoint}")


if __name__ == "__main__":
    main()
