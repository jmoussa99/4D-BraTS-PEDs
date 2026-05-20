Clinical Pipeline Recommendations
=================================

Current Scope
-------------

The first critical deliverable is automatic MRI sequence classification. The
longitudinal segmentation pipeline depends on correctly identifying and
harmonizing T1, T1 contrast-enhanced, T2, and FLAIR sequences across visits.

The initial longitudinal evaluation should use approximately 40 pediatric brain
tumor patients with serial brain MRI. Each patient may have close to 10 exams,
but the first deliverable should focus on:

* baseline/diagnosis imaging,
* one mid-treatment follow-up,
* one end-of-treatment follow-up.

At minimum, the pipeline should generate tumor masks for at least two
longitudinal timepoints per patient. Full-trajectory forecasting is out of scope
for the current milestone.


Ground Truth And Labels
-----------------------

Use manual reference-standard diagnosis-time annotations as true ground truth.
Use pediatric nnU-Net generated masks as pseudo-labels only when manual masks
are unavailable. Reports, slides, and metrics should clearly separate:

* manual reference-standard masks,
* pseudo-label masks,
* model-generated masks for review.

This avoids presenting a downstream result trained or evaluated on automatic
masks as if it were fully manually validated.


Recommended Pipeline
--------------------

1. Create a brain-only manifest and omit spine visits.
2. Run sequence classification for all candidate MRI volumes.
3. Review sequence-classification confidence and selected paths.
4. Build a baseline/mid/end manifest per patient.
5. Generate or attach segmentation masks for selected timepoints.
6. Train/run longitudinal segmentation for observed timepoints.
7. Export volumetric agreement, temporal consistency, overlays, and volume plots.
8. Create a radiology review sheet with Likert/acceptability fields.


Sequence Classification
-----------------------

Sequence classification is a first-class component, not an implementation
detail. For each selected scan, record:

* source path,
* predicted modality,
* confidence,
* vote distribution,
* status/error,
* whether visual/manual review accepts it.

Low-confidence sequence choices should be reviewed before segmentation.


Preprocessing And Harmonization
-------------------------------

Before stacking modalities, every visit should be resampled to a common grid.
The loader already resamples modalities to the visit mask/reference grid. Review
outputs should be inspected for:

* wrong sequence selection,
* left-right or superior-inferior orientation problems,
* shifted masks,
* large volume jumps caused by preprocessing,
* failed scans or poor image quality.


Evaluation
----------

Quantitative evaluation should prioritize:

* volumetric agreement,
* predicted volume over time,
* target/reference volume over time when available,
* temporal consistency of volume and mask behavior.

Dice can be reported for the manually annotated subset, but broad review should
use radiologist visual evaluation with a Likert or acceptable/unacceptable
scale.

Suggested review fields:

* acceptable / borderline / unacceptable,
* Likert score 1-5,
* failure reason,
* reviewer,
* notes.


Deliverables
------------

The current milestone should produce:

* sequence-classification CSV,
* brain-only baseline/mid/end manifest,
* longitudinal segmentation masks for selected timepoints,
* overlay PNGs for radiologist review,
* volumetric CSVs and plots,
* temporal consistency metrics,
* Likert review spreadsheet,
* architecture diagram,
* preprocessing workflow,
* reproducible inference commands.
