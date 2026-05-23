#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether MRISeqClassifier is ready to run.")
    parser.add_argument("--mriseqclassifier-repo", type=Path, default=Path("MRISeqClassifier"))
    args = parser.parse_args()

    repo = args.mriseqclassifier_repo
    checks = [
        (repo.exists(), f"repo exists: {repo}"),
        ((repo / "05_toolkit.py").exists(), "05_toolkit.py exists"),
        ((repo / "02_models" / "best_model").exists(), "02_models/best_model exists"),
    ]

    model_root = repo / "02_models" / "best_model"
    model_files = sorted(model_root.rglob("*mid_best_model.pth")) if model_root.exists() else []
    checks.append((bool(model_files), "at least one *mid_best_model.pth exists"))

    for ok, label in checks:
        print(f"{'OK' if ok else 'MISSING'}  {label}")

    if model_files:
        print("\nModel files:")
        for path in model_files:
            print(f"  {path}")
    else:
        print("\nDownload the pretrained best_model folder from the MRISeqClassifier README Google Drive link.")
        print(f"Place it at: {model_root}")

    if not all(ok for ok, _ in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
