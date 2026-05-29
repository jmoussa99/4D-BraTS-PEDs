#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], skip: bool = False) -> None:
    print("\n" + " ".join(str(part) for part in command))
    if skip:
        print("SKIPPED")
        return
    subprocess.run(command, cwd=ROOT, check=True)


def _manifest_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _mriseqclassifier_ready(repo: Path) -> bool:
    preferred_model_root = repo / "02_models" / "best_model"
    fallback_model_root = repo / "02_models"
    return (repo / "05_toolkit.py").exists() and (
        (preferred_model_root.exists() and any(preferred_model_root.rglob("*mid_best_model.pth")))
        or any(fallback_model_root.glob("*/*mid_best_model.pth"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end pediatric longitudinal MRI pipeline: sequence QC, "
            "baseline/mid/end selection, pseudo-mask QC, optional training, mask "
            "generation, evaluation, and review export."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("trial"))
    parser.add_argument("--source-manifest", type=Path, default=Path("trial_manifest_with_pediatric_masks.csv"))
    parser.add_argument("--brain-manifest", type=Path, default=Path("trial_manifest_brain.csv"))
    parser.add_argument("--selected-manifest", type=Path, default=Path("trial_manifest_baseline_mid_end.csv"))
    parser.add_argument("--qc-manifest", type=Path, default=Path("trial_manifest_baseline_mid_end_qc.csv"))
    parser.add_argument("--mriseqclassifier-repo", type=Path, default=Path("MRISeqClassifier"))
    parser.add_argument("--sequence-qc-output", type=Path, default=Path("runs/sequence_qc.csv"))
    parser.add_argument("--skip-sequence-qc", action="store_true")
    parser.add_argument("--full-sequence-qc", action="store_true", help="Classify every brain image recursively instead of only selected manifest inputs.")
    parser.add_argument("--timepoints", type=int, default=3)
    parser.add_argument("--min-timepoints", type=int, default=2)
    parser.add_argument("--train", action="store_true", help="Train a new longitudinal model during this run.")
    parser.add_argument("--allow-unqc-training", action="store_true", help="Allow training on selected pseudo masks if QC keeps zero visits.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/longitumor_observed_gpu/last.pt"))
    parser.add_argument("--train-output-dir", type=Path, default=Path("runs/longitumor_observed_gpu"))
    parser.add_argument("--mask-output-dir", type=Path, default=Path("runs/longitumor_observed_gpu_masks"))
    parser.add_argument("--generated-manifest", type=Path, default=Path("runs/longitumor_observed_gpu_manifest.csv"))
    parser.add_argument("--eval-output-dir", type=Path, default=Path("runs/longitumor_observed_gpu_eval"))
    parser.add_argument("--review-output-dir", type=Path, default=Path("runs/longitumor_review_cases"))
    parser.add_argument("--pseudo-qc-report", type=Path, default=Path("runs/pseudo_mask_qc.csv"))
    parser.add_argument("--pseudo-qc-mask-dir", type=Path, default=Path("runs/pseudo_masks_qc"))
    parser.add_argument("--min-component-ml", type=float, default=0.05)
    parser.add_argument("--max-components-per-label", type=int, default=3)
    args = parser.parse_args()

    py = sys.executable
    source_manifest = args.source_manifest

    if not source_manifest.exists():
        source_manifest = args.brain_manifest
        _run(
            [
                py,
                "scripts/create_manifest.py",
                "--data-dir",
                str(args.data_dir),
                "--output",
                str(args.brain_manifest),
                "--include-visit-token",
                "brain",
                "--exclude-visit-token",
                "spine",
            ]
        )

    _run(
        [
            py,
            "scripts/select_longitudinal_timepoints.py",
            "--manifest",
            str(source_manifest),
            "--output",
            str(args.selected_manifest),
            "--timepoints",
            str(args.timepoints),
            "--min-timepoints",
            str(args.min_timepoints),
            "--require-mask",
        ]
    )

    if not args.skip_sequence_qc:
        if _mriseqclassifier_ready(args.mriseqclassifier_repo):
            if args.full_sequence_qc:
                _run(
                    [
                        py,
                        "scripts/classify_sequences.py",
                        "--input",
                        str(args.data_dir / "trial" if (args.data_dir / "trial").exists() else args.data_dir),
                        "--mriseqclassifier-repo",
                        str(args.mriseqclassifier_repo),
                        "--output",
                        str(args.sequence_qc_output),
                        "--include-path-token",
                        "brain",
                        "--exclude-path-token",
                        "spine",
                    ]
                )
            else:
                _run(
                    [
                        py,
                        "scripts/classify_manifest_sequences.py",
                        "--manifest",
                        str(args.selected_manifest),
                        "--mriseqclassifier-repo",
                        str(args.mriseqclassifier_repo),
                        "--output",
                        str(args.sequence_qc_output),
                    ]
                )
        else:
            print("\nMRISeqClassifier weights are missing; sequence QC was skipped.")
            print(f"Expected weights under: {args.mriseqclassifier_repo / '02_models' / 'best_model'}")

    _run(
        [
            py,
            "scripts/qc_pseudo_masks.py",
            "--manifest",
            str(args.selected_manifest),
            "--output-manifest",
            str(args.qc_manifest),
            "--output-mask-dir",
            str(args.pseudo_qc_mask_dir),
            "--qc-report",
            str(args.pseudo_qc_report),
            "--min-component-ml",
            str(args.min_component_ml),
            "--max-components-per-label",
            str(args.max_components_per_label),
        ]
    )

    qc_rows = _manifest_rows(args.qc_manifest)
    selected_rows = _manifest_rows(args.selected_manifest)
    training_manifest = args.qc_manifest if qc_rows else args.selected_manifest

    if args.train:
        if qc_rows == 0 and not args.allow_unqc_training:
            raise SystemExit(
                "Pseudo-mask QC kept 0 visits. Refusing to train on failed pseudo labels. "
                "Use --allow-unqc-training only for a clearly labeled weak-baseline experiment."
            )
        _run(
            [
                py,
                "scripts/train_longitudinal.py",
                "--manifest",
                str(training_manifest),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--output-dir",
                str(args.train_output_dir),
                "--device",
                args.device,
            ]
        )
        checkpoint = args.train_output_dir / "last.pt"
    else:
        checkpoint = args.checkpoint
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint does not exist and --train was not set: {checkpoint}")

    _run(
        [
            py,
            "scripts/generate_masks.py",
            "--manifest",
            str(args.selected_manifest),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(args.mask_output_dir),
            "--output-manifest",
            str(args.generated_manifest),
            "--device",
            args.device,
            "--write-modality-space-masks",
            "--write-pseudo-mask-copies",
            "--postprocess",
            "--min-component-ml",
            str(args.min_component_ml),
            "--max-components-per-label",
            str(args.max_components_per_label),
        ]
    )

    _run(
        [
            py,
            "scripts/export_review_cases.py",
            "--manifest",
            str(args.selected_manifest),
            "--pred-mask-dir",
            str(args.mask_output_dir),
            "--output-dir",
            str(args.review_output_dir),
            "--copy-original-pseudo-mask",
        ]
    )

    _run(
        [
            py,
            "scripts/evaluate_longitudinal.py",
            "--manifest",
            str(args.selected_manifest),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(args.eval_output_dir),
            "--device",
            args.device,
            "--postprocess",
            "--min-component-ml",
            str(args.min_component_ml),
            "--max-components-per-label",
            str(args.max_components_per_label),
        ]
    )

    _run(
        [
            py,
            "scripts/create_review_sheet.py",
            "--overlay-dir",
            str(args.eval_output_dir / "overlays"),
            "--output",
            str(args.eval_output_dir / "visual_review.csv"),
        ]
    )

    print("\nPipeline complete.")
    print(f"Selected visits: {selected_rows}")
    print(f"QC-retained visits: {qc_rows}")
    print(f"Candidate masks: {args.mask_output_dir}")
    print(f"ITK-SNAP review folders: {args.review_output_dir}")
    print(f"Metrics and plots: {args.eval_output_dir}")
    print(f"Pseudo-mask QC report: {args.pseudo_qc_report}")


if __name__ == "__main__":
    main()
