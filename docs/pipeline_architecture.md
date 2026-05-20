Pipeline Architecture
=====================

End-To-End Flow
---------------

```mermaid
flowchart TD
    A[Raw clinical MRI folders] --> B[Brain-only visit discovery]
    B --> C[MRI sequence classification]
    C --> D[T1/T1c/T2/FLAIR manifest]
    D --> E[Baseline/mid/end timepoint selection]
    E --> F[Preprocessing and harmonization]
    F --> G[Longitudinal segmentation model]
    G --> H[Mask outputs]
    H --> I[Volume and temporal metrics]
    H --> J[Overlay PNGs]
    I --> K[Radiology review sheet]
    J --> K
```


Preprocessing Workflow
----------------------

```mermaid
flowchart LR
    A[Selected modality paths] --> B[Read NIfTI/MHA]
    B --> C[Choose visit reference grid]
    C --> D[Resample modalities]
    D --> E[Z-score nonzero voxels]
    E --> F[Stack T1/T2/T1c/FLAIR]
    F --> G[Track missing modality availability]
    G --> H[Model input tensor]
```


Model Input And Output
----------------------

Input tensor:

    [batch, time, modality, depth, height, width]

Modality order:

    T1, T2, T1c, FLAIR

Primary output:

    segmentation probabilities [batch, time, label, depth, height, width]

The current milestone uses observed longitudinal segmentation. Future
segmentation forecasting is intentionally excluded from this scope.


Review Outputs
--------------

The reproducible inference pipeline should export:

* predicted masks,
* overlay PNGs,
* per-label and whole-tumor volumes,
* volume-over-time plots,
* temporal consistency metrics,
* radiologist review CSV.
