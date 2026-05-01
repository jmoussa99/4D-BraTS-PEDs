#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.training import train_longitudinal
from longitumor.utils import parse_patch_size, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 longitudinal LongiTumorMamba fine-tuning.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/longitumor_longitudinal"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", default="96,160,160")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lambda-temp", type=float, default=0.1)
    parser.add_argument("--lambda-shape", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    last = train_longitudinal(
        manifest=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=parse_patch_size(args.patch_size),
        learning_rate=args.learning_rate,
        lambda_temp=args.lambda_temp,
        lambda_shape=args.lambda_shape,
        device_name=args.device,
    )
    print(f"Last checkpoint: {last}")


if __name__ == "__main__":
    main()
