#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.training import run_synthetic_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic LongiTumorMamba smoke test.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_synthetic_smoke(args.device)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
