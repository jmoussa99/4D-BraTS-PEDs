#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.training import train_single_timepoint
from longitumor.utils import parse_patch_size, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 single-timepoint LongiTumorMamba training.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/longitumor_single"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", default="96,160,160")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    best = train_single_timepoint(
        manifest=args.manifest,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=parse_patch_size(args.patch_size),
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        device_name=args.device,
    )
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
