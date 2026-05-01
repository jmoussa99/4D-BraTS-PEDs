#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import discover_cases, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a LongiTumorMamba manifest from case folders.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("longitumor_manifest.csv"))
    args = parser.parse_args()

    records = discover_cases(args.data_dir)
    write_manifest(records, args.output)
    labeled = sum(1 for record in records if record.mask_path)
    print(f"Wrote {len(records)} records ({labeled} labeled) to {args.output}")


if __name__ == "__main__":
    main()
