#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a radiology visual review sheet for PNG prediction overlays.")
    parser.add_argument("--overlay-dir", type=Path, required=True, help="Folder containing patient subfolders with PNG overlays.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    overlays = sorted(args.overlay_dir.rglob("*.png"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "overlay_path",
                "patient_id",
                "case_id",
                "acceptability",
                "likert_1_5",
                "failure_reason",
                "reviewer",
                "notes",
            ],
        )
        writer.writeheader()
        for overlay in overlays:
            patient_id = overlay.parent.name
            writer.writerow(
                {
                    "overlay_path": str(overlay),
                    "patient_id": patient_id,
                    "case_id": overlay.stem,
                    "acceptability": "",
                    "likert_1_5": "",
                    "failure_reason": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
    print(f"Wrote {len(overlays)} review rows to {args.output}")


if __name__ == "__main__":
    main()
