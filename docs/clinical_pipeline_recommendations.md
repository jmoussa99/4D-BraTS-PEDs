Clinical Pipeline Recommendations
=================================

This plan reflects the current project goal: demonstrate an end-to-end,
clinically inspectable pediatric brain tumor MRI pipeline before expanding to a
larger server-side cohort.


Primary Goal
------------

Build a working brain-only pipeline that can:

1. select T1, T1c, T2, and FLAIR sequences from raw clinical folders,
2. normalize/resample each visit into a consistent 3D grid,
3. generate longitudinal tumor masks,
4. estimate tumor volume over time,
5. export visual review images for radiologist Likert scoring, and
6. optionally forecast the next-visit segmentation from prior timepoints.

For the June checkpoint, the most important deliverable is an integrated
pipeline with inspectable outputs, not a fully validated Dice benchmark over all
timepoints.


Ground Truth Strategy
---------------------

Separate mask sources clearly:

* Manual reference-standard masks: use these as true supervised labels and as
  the main quality anchor. Diagnosis-time annotated imaging should be treated
  as ground truth where available.
* Pediatric nnU-Net masks: use these as pseudo-labels for bootstrapping,
  pipeline integration, visual review, and weak supervision. Do not present
  downstream evolution metrics from pseudo-labels as if they were fully
  validated ground-truth results.
* Visual Likert review: use radiologist acceptability scoring for broader
  review where full reference standards are not available.

Avoid a hidden "prediction from prediction" claim. If a future model is trained
on pseudo-label masks, report it as pseudo-label-supervised forecasting until a
manual subset is reviewed.


Recommended Default Cohort
--------------------------

Use brain visits only. Spine imaging is a separate metastasis task and should
not be mixed into the brain tumor segmentation pipeline.

Create the brain-only manifest:

    .\.venv-nnunet\Scripts\python.exe scripts\create_manifest.py ^
      --data-dir trial ^
      --output trial_manifest_brain.csv ^
      --include-visit-token brain ^
      --exclude-visit-token spine

Use three to five timepoints per patient for the first longitudinal demo. The
folder day prefix, such as `1921d`, is chronological time in days; it is not an
actual calendar date.


Sequence Selection
------------------

Run the sequence classifier at least on the three available patients and compare
its selected T1/T1c/T2/FLAIR volumes against the diagnosis-time manually selected
series where those are known. Record:

* predicted modality,
* confidence,
* selected file path,
* whether visual/manual review accepts the selection.

Clinical folders include vendor-specific protocols and localizers/scouts, so
sequence selection should be a first-class QC step, not an invisible detail.


Orientation And Resampling QC
-----------------------------

The pipeline should resample modalities to a common visit grid before stacking
channels. This is already implemented in `longitumor.data` for training. Keep
orientation QC visible because anonymization may remove or alter metadata. For
review exports, inspect overlays for:

* left-right or upside-down orientation problems,
* modality mismatch,
* masks shifted relative to tumor,
* inconsistent volume jumps caused by registration/sequence errors.


Training Tracks
---------------

Use two separate tracks:

* Observed longitudinal segmentation:
  `input visits 0..N -> masks for visits 0..N`.
  This tests whether temporal context improves segmentation consistency over
  observed scans.

* Future segmentation forecasting:
  `input visits 0..N-1 -> mask for visit N`.
  This is the actual next-visit prediction task. Treat results as experimental,
  especially if trained from pseudo-label masks.

Do not merge these claims in presentation slides.


Review Outputs
--------------

For broad review, export PNG overlays and a CSV review sheet. Dice is useful for
the small manually annotated subset, but the larger trial review should use a
Likert/acceptability scale:

* acceptable,
* borderline,
* unacceptable,
* failure reason.

Recommended failure reasons:

* wrong sequence,
* wrong orientation,
* missed tumor,
* oversegmentation,
* shifted mask,
* poor image quality,
* uncertain anatomy.


Minimum Demo Checklist
----------------------

Before the next update, produce:

1. brain-only manifest,
2. sequence classifier CSV for the three patients,
3. pediatric nnU-Net pseudo-masks or manual masks attached to manifest,
4. observed longitudinal segmentation checkpoint,
5. future segmentation checkpoint,
6. overlay PNGs for observed and future predictions,
7. volume-over-time CSV/plot,
8. visual review CSV for radiologist scoring.
