Longitudinal Pediatric Brain Tumor MRI Pipeline
==============================================

This repository now focuses on the current milestone:

1. automatic MRI sequence classification,
2. brain-only longitudinal timepoint selection,
3. tumor segmentation across selected longitudinal visits,
4. volumetric and temporal consistency evaluation,
5. radiologist visual review with Likert/acceptability scoring,
6. reproducible inference and review outputs.

Future segmentation forecasting is intentionally out of scope for this
milestone.


Data Layout
-----------

Expected trial layout:

    trial/trial/{patient}/{visit}/{series}/*.nii.gz

Manifest columns:

    patient_id,visit_id,delta_t,t1,t2,t1c,flair,mask,previous_mask

Modality order inside this codebase:

    T1, T2, T1c, FLAIR

The pediatric nnU-Net model expects:

    FLAIR, T1, T1C, T2


Setup
-----

Use the CUDA environment for the current GPU workstation:

    .\.venv-nnunet\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

Install dependencies if needed:

    .\.venv-nnunet\Scripts\python.exe -m pip install nnunetv2 SimpleITK nibabel tqdm matplotlib scipy scikit-image


Step 1: Brain-Only Manifest
---------------------------

Create a brain-only manifest and omit spine visits:

    .\.venv-nnunet\Scripts\python.exe scripts\create_manifest.py ^
      --data-dir trial ^
      --output trial_manifest_brain.csv ^
      --include-visit-token brain ^
      --exclude-visit-token spine


Step 2: Sequence Classification QC
----------------------------------

Run MRISeqClassifier on candidate images. This is the first critical component
because downstream segmentation depends on correct sequence identification.

    .\.venv-nnunet\Scripts\python.exe scripts\classify_sequences.py ^
      --input trial\trial ^
      --mriseqclassifier-repo MRISeqClassifier ^
      --output runs\sequence_qc.csv ^
      --include-path-token brain ^
      --exclude-path-token spine

To create a manifest using MRISeqClassifier predictions:

    .\.venv-nnunet\Scripts\python.exe scripts\create_manifest.py ^
      --data-dir trial ^
      --output trial_manifest_brain_classifier.csv ^
      --include-visit-token brain ^
      --exclude-visit-token spine ^
      --mriseqclassifier-repo MRISeqClassifier ^
      --classifier-python .\.venv-nnunet\Scripts\python.exe


Step 3: Baseline/Mid/End Cohort
-------------------------------

Select clinically meaningful longitudinal visits per patient. For the first
deliverable, use baseline/diagnosis, mid-treatment, and end-of-treatment when
available. At minimum, keep patients with at least two selected timepoints.

    .\.venv-nnunet\Scripts\python.exe scripts\select_longitudinal_timepoints.py ^
      --manifest trial_manifest_with_pediatric_masks.csv ^
      --output trial_manifest_baseline_mid_end.csv ^
      --timepoints 3 ^
      --min-timepoints 2 ^
      --require-mask


Step 4: Segmentation Masks
--------------------------

Generate pediatric nnU-Net pseudo-masks when manual masks are unavailable:

    .\.venv-nnunet\Scripts\python.exe scripts\generate_masks.py ^
      --manifest trial_manifest_baseline_mid_end.csv ^
      --pediatric-model-repo pediatric_model ^
      --output-dir trial_pediatric_masks_selected ^
      --output-manifest trial_manifest_baseline_mid_end_masks.csv ^
      --device 0

Manual reference-standard diagnosis-time masks should be treated as true
ground truth. Pediatric nnU-Net masks are pseudo-labels unless manually reviewed.


Step 5: Longitudinal Segmentation
---------------------------------

Train observed longitudinal segmentation:

    .\.venv-nnunet\Scripts\python.exe scripts\train_longitudinal.py ^
      --manifest trial_manifest_baseline_mid_end_masks.csv ^
      --epochs 100 ^
      --batch-size 1 ^
      --output-dir runs\longitumor_observed_gpu ^
      --device cuda

Generate model masks with the trained checkpoint:

    .\.venv-nnunet\Scripts\python.exe scripts\generate_masks.py ^
      --manifest trial_manifest_baseline_mid_end_masks.csv ^
      --checkpoint runs\longitumor_observed_gpu\last.pt ^
      --output-dir runs\longitumor_observed_gpu_masks ^
      --output-manifest runs\longitumor_observed_gpu_manifest.csv ^
      --device cuda


Step 6: Evaluation And Review Outputs
-------------------------------------

Export overlays, volumetric agreement, temporal consistency, and volume plots:

    .\.venv-nnunet\Scripts\python.exe scripts\evaluate_longitudinal.py ^
      --manifest trial_manifest_baseline_mid_end_masks.csv ^
      --checkpoint runs\longitumor_observed_gpu\last.pt ^
      --output-dir runs\longitumor_observed_gpu_eval ^
      --device cuda

Create a radiology Likert review sheet:

    .\.venv-nnunet\Scripts\python.exe scripts\create_review_sheet.py ^
      --overlay-dir runs\longitumor_observed_gpu_eval\overlays ^
      --output runs\longitumor_observed_gpu_eval\visual_review.csv

Main review outputs:

    runs\longitumor_observed_gpu_eval\metrics.csv
    runs\longitumor_observed_gpu_eval\temporal_consistency.csv
    runs\longitumor_observed_gpu_eval\metric_trends.png
    runs\longitumor_observed_gpu_eval\volume_trends.png
    runs\longitumor_observed_gpu_eval\visual_review.csv


Architecture And Workflow Docs
------------------------------

See:

    docs/clinical_pipeline_recommendations.md
    docs/pipeline_architecture.md


Version Control Notes
---------------------

Generated data, masks, checkpoints, manifests, archives, and medical image files
are intentionally ignored by git. Keep source code, tests, and documentation in
git; keep datasets and model outputs outside git.
