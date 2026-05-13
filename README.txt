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
      train_single_timepoint.py   Single-timepoint segmentation training
      train_longitudinal.py       Longitudinal model training
      train_sequence_classifier.py
      classify_sequences.py
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

The discovery code handles:

* nested `{patient}/{visit}/{series}` folders,
* wrapper folders such as `trial/trial`,
* visit IDs that start with day counts such as `1921d_B_brain_12h46m`,
* missing modalities,
* modality selection from clinical series names,
* preference for larger/full series over tiny one-slice files.


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
