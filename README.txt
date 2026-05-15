LongiTumor 4D-BraTS PEDs
========================

This repository contains utilities for longitudinal pediatric brain tumor MRI
experiments. It can discover multimodal visits from nested clinical trial data,
build longitudinal manifests, generate pediatric tumor masks with the
NUBagciLab nnU-Net models, and train/evaluate a lightweight longitudinal 4D MRI
segmentation and tumor evolution model.

The current trial layout is:

    trial/trial/{patient}/{visit}/{series}/*.nii.gz

The manifest format used by the scripts is:

    patient_id,visit_id,delta_t,t1,t2,t1c,flair,mask,previous_mask

Modality order inside the project is T1, T2, T1c, FLAIR. The external pediatric
nnU-Net model is staged in its expected order: FLAIR, T1, T1C, T2.


Architecture
------------

The core model is `LongiTumorMamba` in `longitumor/models.py`.

Input:

    x: [batch, time, modality, depth, height, width]

Each visit can have missing modalities. `longitumor/data.py` records modality
availability, and the model concatenates image channels with availability maps
so it can distinguish a missing scan from a real zero-valued image.

Main components:

* Modality dropout: randomly drops modalities during training to improve
  robustness to incomplete clinical scans.
* Local 3D encoder: residual 3D convolution blocks extract per-visit features.
* Temporal tetra mixer: mixes bottleneck features across flattened spatial
  tokens, reverse tokens, time, and depth. If `mamba-ssm` is installed, it uses
  Mamba blocks; otherwise it falls back to a Conv1d sequence mixer.
* Optional bottleneck attention: enabled from config when needed.
* Optional shape memory branch: can encode previous masks and fuse them into the
  bottleneck.
* 3D decoder: upsamples with skip connections and predicts tumor label logits.
* Evolution head: produces longitudinal response/risk outputs from trajectory
  embeddings and segmentation-derived volume summaries.

The model returns logits, probabilities, trajectory embeddings, and evolution
outputs through `LongiTumorMambaOutput`.


Repository Layout
-----------------

    longitumor/
      data.py                 Dataset discovery, manifest IO, MRI loading
      models.py               LongiTumorMamba and supporting blocks
      training.py             Training utilities
      inference.py            Checkpoint and pediatric nnU-Net mask generation
      evaluation.py           Metrics/evaluation helpers
      sequence_classifier.py  Optional MRI sequence classifier wrapper

    scripts/
      create_manifest.py          Discover visits and write a manifest
      generate_masks.py           Generate masks from a manifest
      evaluate_longitudinal.py    Export observed-visit overlays, metrics, volumes
      train_single_timepoint.py   Single-timepoint segmentation training
      train_longitudinal.py       Longitudinal model training
      train_future_segmentation.py
      predict_future_segmentation.py
      train_sequence_classifier.py
      classify_sequences.py
      create_review_sheet.py
      smoke_test.py

    tests/
      test_synthetic.py       Fast synthetic coverage for discovery/model code

Generated data, masks, model checkpoints, archives, and CSV manifests are
ignored by git.


Setup
-----

Create the main Python environment:

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt

For pediatric nnU-Net inference, create a separate environment. On a CUDA
Windows machine, install a CUDA PyTorch wheel before running inference:

    py -3.11 -m venv .venv-nnunet
    .\.venv-nnunet\Scripts\python.exe -m pip install --upgrade pip
    .\.venv-nnunet\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130

Then install nnU-Net dependencies used by the pediatric model:

    .\.venv-nnunet\Scripts\python.exe -m pip install nnunetv2 SimpleITK nibabel tqdm

Verify CUDA:

    .\.venv-nnunet\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"


Pediatric Pretrained Model Setup
--------------------------------

Clone the NUBagciLab (https://github.com/NUBagciLab/Pediatric-Brain-Tumor-Segmentation-Model) pediatric model repository into `pediatric_model/`.

Install the downloaded pretrained archives so the repo contains:

    pediatric_model/data/nnUNet_results/Dataset106_WTPED24/...
    pediatric_model/data/nnUNet_results/Dataset107_3LabelPED24/...

The inference wrapper expects the repository's conversion script at:

    pediatric_model/postProcessing/conversion.py

The pipeline runs:

1. `Dataset106_WTPED24` for whole tumor.
2. `Dataset107_3LabelPED24` for 3-label prediction.
3. `postProcessing/conversion.py` to combine/relabeled outputs.


Create A Manifest
-----------------

For the trial dataset:

    .\.venv\Scripts\python.exe scripts\create_manifest.py --data-dir trial --output trial_manifest.csv

For the recommended brain-only clinical demo cohort:

    .\.venv-nnunet\Scripts\python.exe scripts\create_manifest.py ^
      --data-dir trial ^
      --output trial_manifest_brain.csv ^
      --include-visit-token brain ^
      --exclude-visit-token spine

The discovery code handles:

* nested `{patient}/{visit}/{series}` folders,
* wrapper folders such as `trial/trial`,
* visit IDs that start with day counts such as `1921d_B_brain_12h46m`,
* missing modalities,
* modality selection from clinical series names,
* preference for larger/full series over tiny one-slice files.

Spine imaging should stay out of the default brain tumor pipeline. It is useful
for metastasis review, but spinal tumor detection/segmentation is a separate
problem.


Sequence Selection QC
---------------------

Run the MRI sequence classifier on the available patients before relying on
automatic T1/T1c/T2/FLAIR selection:

    .\.venv-nnunet\Scripts\python.exe scripts\classify_sequences.py ^
      --checkpoint sequence_classifier_checkpoint.pt ^
      --input trial\trial ^
      --output runs\sequence_qc.csv ^
      --device cuda

Compare the selected sequence paths against any manually selected
diagnosis-time reference sequences. Low-confidence or wrong sequence choices
should be flagged for visual review.


Generate Pediatric Masks
------------------------

Run a one-visit smoke test first:

    .\.venv-nnunet\Scripts\python.exe scripts\generate_masks.py ^
      --manifest trial_manifest.csv ^
      --pediatric-model-repo pediatric_model ^
      --output-dir trial_pediatric_masks_smoke ^
      --output-manifest trial_manifest_with_pediatric_masks_smoke.csv ^
      --device 0 ^
      --limit 1

Then run the full manifest:

    .\.venv-nnunet\Scripts\python.exe scripts\generate_masks.py ^
      --manifest trial_manifest.csv ^
      --pediatric-model-repo pediatric_model ^
      --output-dir trial_pediatric_masks ^
      --output-manifest trial_manifest_with_pediatric_masks.csv ^
      --device 0

Use `--device cpu` only for debugging. Full CPU inference can be very slow.


Training And Tests
------------------

Run tests:

    .\.venv\Scripts\python.exe -m pytest -q

Train entry points are under `scripts/`. Use a manifest with populated `mask`
and `previous_mask` columns for supervised longitudinal training.

Example:

    .\.venv\Scripts\python.exe scripts\train_longitudinal.py --manifest trial_manifest_with_pediatric_masks.csv

Exact training arguments may vary by experiment; run any script with `--help`
to see its options.


Observed And Future Longitudinal Outputs
----------------------------------------

There are two separate modeling tracks:

1. Observed longitudinal segmentation:

       input visits 0..N -> masks for visits 0..N

   This uses temporal context to segment observed scans.

2. Future segmentation forecasting:

       input visits 0..N-1 -> mask for visit N

   This is the next-visit tumor mask prediction task.

Train the observed longitudinal model on GPU:

    .\.venv-nnunet\Scripts\python.exe scripts\train_longitudinal.py ^
      --manifest trial_manifest_with_pediatric_masks.csv ^
      --epochs 20 ^
      --batch-size 1 ^
      --output-dir runs\longitumor_longitudinal_gpu ^
      --device cuda

Export observed-visit overlays, metrics, and volume plots:

    .\.venv-nnunet\Scripts\python.exe scripts\evaluate_longitudinal.py ^
      --manifest trial_manifest_with_pediatric_masks.csv ^
      --checkpoint runs\longitumor_longitudinal_gpu\last.pt ^
      --output-dir runs\longitumor_longitudinal_gpu_eval ^
      --device cuda

Train the future segmentation model:

    .\.venv-nnunet\Scripts\python.exe scripts\train_future_segmentation.py ^
      --manifest trial_manifest_with_pediatric_masks.csv ^
      --epochs 20 ^
      --batch-size 1 ^
      --input-timepoints 3 ^
      --output-dir runs\longitumor_future ^
      --device cuda

Evaluate known next-visit forecasting windows:

    .\.venv-nnunet\Scripts\python.exe scripts\predict_future_segmentation.py ^
      --manifest trial_manifest_with_pediatric_masks.csv ^
      --checkpoint runs\longitumor_future\last.pt ^
      --output-dir runs\longitumor_future_predictions ^
      --input-timepoints 3 ^
      --device cuda

Forecast after each patient's latest visit:

    .\.venv-nnunet\Scripts\python.exe scripts\predict_future_segmentation.py ^
      --manifest trial_manifest_with_pediatric_masks.csv ^
      --checkpoint runs\longitumor_future\last.pt ^
      --output-dir runs\longitumor_future_latest ^
      --input-timepoints 3 ^
      --device cuda ^
      --latest

Use `--patch-size full` with `predict_future_segmentation.py` to write
full-volume NIfTI forecast masks in addition to PNG overlays.


Visual Review
-------------

For broad clinical review, use exported overlays and a Likert/acceptability
sheet. Dice scores should be reserved for the manually annotated subset or a
small curated sample.

Create a review CSV from exported PNG overlays:

    .\.venv-nnunet\Scripts\python.exe scripts\create_review_sheet.py ^
      --overlay-dir runs\longitumor_longitudinal_gpu_eval\overlays ^
      --output runs\longitumor_longitudinal_gpu_eval\visual_review.csv

Suggested fields are included in the CSV: acceptability, Likert score, failure
reason, reviewer, and notes.


Clinical Pipeline Notes
-----------------------

See `docs/clinical_pipeline_recommendations.md` for the current recommended
demo plan. Key points:

* manual reference-standard masks are true labels,
* pediatric nnU-Net masks are pseudo-labels unless manually reviewed,
* brain-only visits should be the default cohort,
* orientation and sequence-selection QC must be visually checked,
* volume-over-time trends are a primary review output.


Version Control Notes
---------------------

The repo intentionally ignores raw data and generated artifacts:

* `trial/`
* `trial_pediatric_masks*/`
* `nnunet_raw/`, `nnunet_preprocessed/`, `nnunet_results/`
* `pediatric_model/data/`
* CSV manifests
* zip archives
* NIfTI/DICOM/MHA medical imaging files
* checkpoints and model weights

Keep code, tests, and small documentation files in git. Keep datasets,
pretrained checkpoints, generated masks, and experiment outputs outside git.


Data Attribution
----------------

Some original brain tumor image data referenced by this project were obtained
from the MICCAI 2012 Challenge on Multimodal Brain Tumor Segmentation
(BRATS2012), organized by B. Menze, A. Jakab, S. Bauer, M. Reyes, M. Prastawa,
and K. Van Leemput. That challenge database contains fully anonymized images
from ETH Zurich, University of Bern, University of Debrecen, and University of
Utah.
