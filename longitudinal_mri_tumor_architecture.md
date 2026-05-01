# Longitudinal MRI Tumor Architecture Spec

## Goal

Design a longitudinal brain MRI model that takes multiparametric MRI sequences across visits and produces temporally consistent tumor segmentations. The segmentations are then converted into quantitative tumor-evolution features for growth, treatment response, progression, and survival modeling.

The proposed architecture is named `LongiTumorMamba`.

## Paper-Derived Design Principles

This design synthesizes four ideas from the provided papers:

- **Robust missing-sequence segmentation**: The pediatric brain tumor paper showed that modality dropout during segmentation training was more robust than synthesis, copy substitution, or zero-filled inputs when FLAIR or T1-weighted sequences were missing. `LongiTumorMamba` uses modality dropout as the primary missing-modality strategy rather than relying on image synthesis as a required preprocessing step.
- **4D spatio-temporal modeling**: OmniMamba4D treats longitudinal imaging as a 4D input, using Mamba blocks to capture spatial and temporal dependencies efficiently. `LongiTumorMamba` adopts this principle for longitudinal MRI, processing tensors shaped as `(B, T, M, D, H, W)`.
- **Previous-mask shape guidance**: MambaX-Net improves longitudinal segmentation by conditioning the current segmentation on the previous scan and previous mask. `LongiTumorMamba` adds a shape-memory branch that uses prior predicted or manual masks to stabilize boundaries and reduce temporal flicker.
- **Efficient hybrid staging**: SegMaFormer uses Mamba layers where token sequences are long and attention where tokens are compact. `LongiTumorMamba` follows this staged design to keep high-resolution processing efficient while still using attention at low resolution for global tumor and anatomy context.

## Input and Output

### Input

The primary input is a longitudinal multiparametric MRI tensor:

```text
X: (B, T, M, D, H, W)
```

Where:

- `B` is batch size.
- `T` is the number of timepoints or visits.
- `M` is the MRI sequence count: `T1`, `T2`, `T1c`, and `FLAIR`.
- `D`, `H`, and `W` are the 3D volume dimensions after registration, resampling, and cropping.

The model also accepts optional metadata:

```text
delta_t: (B, T)
available_modalities: (B, T, M)
clinical_covariates: age, sex, treatment group, diagnosis, molecular markers when available
previous_masks: (B, T, 4, D, H, W)
```

`delta_t` is important because clinical visits are usually irregularly spaced. The model should learn tumor evolution per unit time, not only per index in the scan sequence.

### Output

The model outputs:

```text
Y_hat: (B, T, 4, D, H, W)
trajectory_embedding: (B, T, E)
evolution_outputs: growth rate, response state, progression risk, survival risk
```

The segmentation output is a 3D tumor volume with four modality-aligned channels:

- `Y_hat[:, :, 0]`: tumor segmentation volume aligned to `T1`
- `Y_hat[:, :, 1]`: tumor segmentation volume aligned to `T2`
- `Y_hat[:, :, 2]`: tumor segmentation volume aligned to `T1c`
- `Y_hat[:, :, 3]`: tumor segmentation volume aligned to `FLAIR`

For a single patient at a single timepoint, the segmentation can be represented as `(4, D, H, W)`. Each channel can store either a binary tumor mask or a tumor probability map for that MRI sequence. A final fused tumor volume can be obtained by averaging, voting, or applying a learned fusion layer across the four modality-aligned channels.

## Architecture Overview

`LongiTumorMamba` has six stages:

1. **Longitudinal MRI preprocessing**
2. **Modality-aware input encoding**
3. **Shared 3D local encoder**
4. **Spatio-temporal Mamba encoder**
5. **Shape-memory and temporal consistency decoder**
6. **Tumor-evolution modeling head**

```mermaid
flowchart TD
    rawMri["Longitudinal MRI visits"] --> preprocess["Register, resample, normalize, crop"]
    preprocess --> inputTensor["Tensor X: B,T,M,D,H,W"]
    inputTensor --> modalityGate["Availability mask and modality dropout"]
    modalityGate --> localEncoder["Shared 3D local encoder"]
    localEncoder --> temporalMamba["Spatio-temporal Mamba encoder"]
    temporalMamba --> lowResAttention["Compact cross-time attention"]
    previousMasks["Previous masks or prior predictions"] --> shapeMemory["Shape-memory encoder"]
    shapeMemory --> decoder["Temporal consistency decoder"]
    lowResAttention --> decoder
    localEncoder --> decoder
    decoder --> segMasks["Per-timepoint tumor masks"]
    segMasks --> featureExtractor["Volume, radiomics, morphology"]
    temporalMamba --> latentTrajectory["Latent trajectory embeddings"]
    featureExtractor --> evolutionHead["Tumor-evolution model"]
    latentTrajectory --> evolutionHead
    clinicalData["Clinical covariates"] --> evolutionHead
    evolutionHead --> outputs["Growth, response, progression, survival risk"]
```

## Core Modules

### 1. Longitudinal MRI Preprocessing

Preprocessing should produce aligned, intensity-normalized longitudinal volumes while preserving clinically meaningful tumor change.

Recommended pipeline:

1. Convert DICOM to NIfTI and organize by patient, visit, and sequence.
2. Register sequences within each visit to a reference sequence, usually T1c or T2.
3. Register visits to a baseline or atlas space using rigid or affine registration first.
4. Use deformable registration cautiously. It can improve alignment, but it may also hide real tumor growth or shrinkage if applied too aggressively.
5. Resample to a common spacing, such as 1 mm isotropic when feasible.
6. Skull-strip if the chosen backbone expects it.
7. Z-score normalize nonzero voxels per sequence and per visit.
8. Crop around brain or tumor region for training; preserve full-volume inference through sliding windows.

Store an `available_modalities` mask for every visit. Missing or corrupted sequences should be explicitly represented instead of silently replaced.

### 2. Modality-Aware Input Encoding

The model receives the available MRI sequences plus an availability mask. During training, it randomly drops selected sequences to simulate real clinical missingness.

Recommended dropout policy:

```text
For each sample and timepoint:
  Drop FLAIR with probability p_flair.
  Drop T1 with probability p_t1.
  Optionally drop T2 or T1c at lower probabilities.
  Replace dropped sequences with zeros.
  Keep the availability mask as a separate conditioning signal.
```

Initial dropout values:

- `p_flair = 0.4`
- `p_t1 = 0.4`
- `p_t2 = 0.1`
- `p_t1c = 0.1`

The pediatric brain tumor paper found `p = 0.4` effective for FLAIR and T1-weighted dropout. Extending lower-probability dropout to T2 and T1c makes the model more robust without overtraining on unlikely missingness patterns.

The input encoder should concatenate image channels and learned modality embeddings:

```text
z_t = Conv3D([MRI_t, availability_mask_t, modality_embeddings])
```

This lets the network distinguish a true zero-valued image region from a missing sequence filled with zeros.

### 3. Shared 3D Local Encoder

Use a shared 3D encoder across timepoints. This preserves the strong spatial inductive bias of nnU-Net-like models while keeping the temporal modeling modular.

Implementation concept:

```text
X: (B, T, M, D, H, W)
reshape to (B*T, M, D, H, W)
apply shared 3D encoder
reshape features back to (B, T, C_l, D_l, H_l, W_l)
```

The encoder should produce multiscale features:

```text
F1: high resolution, local boundary detail
F2: mid resolution, modality-specific tumor context
F3: low resolution, whole tumor context
F4: bottleneck, global anatomy and trajectory context
```

A good first implementation can use nnU-Net-style convolutional blocks with residual connections and instance normalization.

### 4. Spatio-Temporal Mamba Encoder

The spatio-temporal encoder is the main longitudinal module. It extends 3D feature maps across time and models dependencies along:

- Forward spatial token order
- Reverse spatial token order
- Inter-slice depth order
- Temporal visit order

This follows OmniMamba4D's tetra-oriented idea, adapted to multiparametric brain MRI.

For each scale `l`, reshape features:

```text
F_l: (B, T, C_l, D_l, H_l, W_l)
tokens_l: (B, T * D_l * H_l * W_l, C_l)
```

Then apply a `TemporalTetraMambaBlock`:

```text
F_out = F_in
F_out += Mamba_forward(tokens)
F_out += Mamba_reverse(tokens)
F_out += Mamba_depth(tokens)
F_out += Mamba_time(tokens)
F_out = MLP(Norm(F_out)) + F_out
```

For memory efficiency:

- Use Mamba blocks at higher-resolution feature scales.
- Use compact cross-time attention only at the bottleneck where token length is much smaller.
- Use gradient checkpointing for long sequences.
- Train on cropped patches and infer with sliding windows.

### 5. Shape-Memory Branch

The shape-memory branch stabilizes segmentation across visits by encoding prior masks. It is inspired by MambaX-Net's shape extractor, but adapted for tumor segmentation.

Inputs:

```text
M_prev: previous manual mask, pseudo-label, or model prediction
I_prev: previous MRI visit
I_curr: current MRI visit
```

The branch computes:

```text
S_prev = ShapeEncoder3D(M_prev)
F_prev = SharedEncoder(I_prev)
F_curr = SharedEncoder(I_curr)
F_fused = MambaCrossAttention(F_curr, F_prev + S_prev)
```

Use this branch in two modes:

- **Joint 4D mode**: all visits are available, so the Mamba encoder predicts all segmentations together.
- **Sequential mode**: only the current and prior visits are used, so previous masks guide the current segmentation.

Sequential mode is important for real deployment because patients arrive visit by visit. It also helps when some patients have only two scans.

### 6. Temporal Consistency Decoder

The decoder reconstructs one segmentation per timepoint. It combines:

- Same-timepoint skip connections for boundary detail.
- Cross-time Mamba features for temporal context.
- Shape-memory features from previous masks.

Temporal consistency should be encouraged but not forced. Tumors can genuinely grow, shrink, disappear, or transform after treatment.

The decoder should learn consistency through soft constraints:

- Penalize impossible high-frequency flicker.
- Preserve new enhancing or non-enhancing regions when supported by image evidence.
- Allow large changes when `delta_t` is large or treatment status changes.

Output:

```text
Y_hat_t = Decoder(F_t, F_temporal_t, S_prev_t)
```

### 7. Tumor-Evolution Modeling Head

After segmentation, compute per-timepoint quantitative features:

```text
V_t1_t
V_t2_t
V_t1c_t
V_flair_t
V_fused_t
surface_area_t
centroid_t
compactness_t
intensity_summary_t per sequence
radiomics_t optional
latent_embedding_t from Mamba bottleneck
```

Convert these into temporal features:

```text
absolute_volume_change = V_t - V_t_minus_1
relative_volume_change = (V_t - V_t_minus_1) / max(V_t_minus_1, epsilon)
monthly_growth_rate = relative_volume_change / delta_months
modality_volume_delta = modality_volume_fraction_t - modality_volume_fraction_t_minus_1
centroid_shift = distance(centroid_t, centroid_t_minus_1)
```

Evolution head options:

- **Interpretable baseline**: time-varying Cox model using tumor volumes and clinical covariates.
- **Neural temporal model**: GRU, temporal Mamba, or Transformer over visit-level embeddings.
- **Hybrid model**: Cox-compatible risk head using both interpretable volume features and learned trajectory embeddings.

Recommended first version:

```text
risk_t = CoxHead([volumes_t, growth_rates_t, latent_embedding_t, clinical_covariates])
response_t = MLP([volumes_t, growth_rates_t, latent_embedding_t])
```

This keeps the evolution model clinically interpretable while still benefiting from learned imaging features.

## Training Strategy

### Stage 1: Single-Timepoint Segmentation Pretraining

Train the 3D local encoder and decoder on all available labeled scans, treating each timepoint independently.

Objective:

```text
L_seg = DiceCE(Y_hat_t, Y_t)
```

Use nnU-Net-style training:

- Patch sampling around tumor and background.
- Deep supervision at decoder scales.
- Strong spatial augmentation.
- Intensity augmentation per sequence.
- Modality dropout during training.

This gives the model a strong segmentation backbone before learning temporal behavior.

### Stage 2: Longitudinal 4D Fine-Tuning

Fine-tune with sequences of visits from each patient.

Objective:

```text
L_total =
  L_seg
  + lambda_temp * L_temporal
  + lambda_shape * L_shape
  + lambda_evo * L_evolution
```

Recommended initial weights:

```text
lambda_temp = 0.1
lambda_shape = 0.1
lambda_evo = 0.2
```

Tune these on validation data. If segmentations become overly smooth and miss true progression, reduce `lambda_temp`.

### Segmentation Loss

Use Dice plus cross-entropy as the default:

```text
L_seg = L_dice + L_ce
```

For small or irregular tumor volumes, add focal Tversky:

```text
L_seg = L_dice + L_ce + lambda_ft * L_focal_tversky
```

This is useful for small, irregular, or low-contrast tumor regions.

### Temporal Consistency Loss

The goal is to discourage implausible mask flicker without suppressing true biological change.

Use a soft warped consistency term:

```text
L_temporal = mean_t DiceDistance(Y_hat_t, Warp(Y_hat_t_minus_1))
```

Weight by visit interval and image evidence:

```text
weight_t = exp(-delta_months_t / tau) * image_similarity_t
```

Large time gaps should impose weaker consistency. Strong image changes should also weaken consistency.

### Shape-Memory Loss

When previous masks are available:

```text
L_shape = BoundaryDistance(Y_hat_t, ShapeGuidedPrediction_t)
```

This can be implemented as a boundary loss, Hausdorff-style loss, or signed distance transform loss. It should mainly improve boundaries and prevent sudden implausible shape jumps.

### Evolution Loss

Choose the loss based on available labels:

- For survival: negative partial log-likelihood from a Cox model.
- For progression labels: binary cross-entropy or focal loss.
- For response categories: cross-entropy.
- For future tumor volume prediction: smooth L1 or Gaussian negative log-likelihood.

Example:

```text
L_evolution = L_cox + L_response + L_volume_forecast
```

If clinical endpoints are not yet available, train the segmentation model first and compute evolution features offline.

### Pseudo-Label and Self-Training Option

If expert labels are sparse:

1. Train a robust nnU-Net or `LongiTumorMamba` single-timepoint model on labeled data.
2. Generate pseudo-labels for unlabeled longitudinal scans.
3. Estimate uncertainty with test-time augmentation or model ensembling.
4. Fine-tune the longitudinal model using high-confidence pseudo-labels.
5. Down-weight noisy pseudo-labels in the segmentation loss.

Avoid blindly adding all pseudo-labels. The MambaX-Net paper showed that dual-scan models can degrade when noisy pseudo-label volume increases.

## Evaluation Plan

### Segmentation Metrics

Report per class and aggregate:

- Dice score
- Hausdorff distance at 95th percentile
- Average surface distance
- Sensitivity and precision
- Volume similarity

Evaluate both complete and missing-sequence settings:

- All sequences available
- Missing FLAIR
- Missing T1
- Missing FLAIR and T1
- Artifact-corrupted sequence replaced by zero and marked unavailable

### Temporal Consistency Metrics

Add metrics that specifically test longitudinal behavior:

- Volume trajectory smoothness adjusted by `delta_t`
- Mask flicker rate after registration
- New-lesion or disappearing-lesion detection accuracy
- Boundary displacement consistency
- Agreement of predicted growth direction with measured tumor volume change

Do not optimize only for smoothness. A model that never changes can look temporally consistent while missing true progression.

### Tumor-Evolution Metrics

For tumor evolution:

- Monthly absolute and relative volume growth error
- Progression prediction AUROC or AUPRC
- C-index for survival or time-to-progression
- Calibration of risk predictions
- Kaplan-Meier separation for high-risk and low-risk groups
- Agreement with clinical response categories when available

### Ablation Studies

Run these ablations to validate each design choice:

- No modality dropout
- No temporal Mamba, independent 3D segmentation only
- No shape-memory branch
- No bottleneck attention
- No temporal consistency loss
- Sequential mode versus joint 4D mode
- With and without clinical covariates in evolution modeling

## Deployment Modes

### Retrospective Batch Mode

Use when all visits are available:

```text
Input: all visits for one patient
Output: all segmentations and full tumor trajectory
```

This should give the most temporally consistent segmentation because the model sees the full sequence.

### Prospective Clinical Mode

Use when only current and previous visits are available:

```text
Input: previous MRI, previous mask, current MRI
Output: current segmentation and updated tumor-evolution estimate
```

This mode supports real clinical follow-up. It should store each predicted mask for use as the prior mask at the next visit, ideally with uncertainty estimates.

## Suggested Implementation Structure

When converting this design into code, a clean PyTorch structure would be:

```text
models/
  longi_tumor_mamba.py
  modules/
    modality_dropout.py
    local_3d_encoder.py
    temporal_tetra_mamba.py
    shape_memory.py
    temporal_decoder.py
    evolution_head.py
training/
  losses.py
  train_single_timepoint.py
  train_longitudinal.py
  pseudo_label.py
evaluation/
  segmentation_metrics.py
  temporal_metrics.py
  evolution_metrics.py
```

The first implementation should favor correctness and reproducibility over architectural complexity:

1. Start with an nnU-Net-like 3D segmentation baseline.
2. Add modality dropout and availability masks.
3. Add temporal Mamba at bottleneck only.
4. Add multiscale temporal Mamba after the baseline is stable.
5. Add shape-memory conditioning.
6. Add the tumor-evolution head last.

## Practical Defaults

Recommended starting configuration:

```text
Input patch: 96 x 160 x 160 or hardware-dependent nnU-Net patch
Sequences: T1, T2, T1c, FLAIR
Timepoints: 2 to 4 per patient during training
Optimizer: AdamW or SGD with nnU-Net schedule
Batch size: 1 to 2 longitudinal samples
Mixed precision: enabled
Backbone: residual nnU-Net-style 3D encoder and decoder
Temporal block: Mamba at bottleneck, then expand to multiscale
Missing modality handling: modality dropout plus availability mask
Primary segmentation loss: DiceCE
Small-region loss: optional focal Tversky
Evolution baseline: time-varying Cox model from predicted volumes
```

## Risks and Mitigations

- **Registration can hide tumor evolution**: Prefer rigid or affine alignment for temporal consistency, and validate deformable registration carefully.
- **Temporal smoothing can suppress true progression**: Make consistency losses interval-aware and image-aware.
- **Noisy pseudo-labels can degrade longitudinal models**: Use uncertainty filtering and loss down-weighting.
- **Mamba complexity can exceed available data**: Add temporal modules progressively and keep a strong nnU-Net baseline.
- **Missing-modality behavior can be brittle**: Always include an availability mask and explicitly evaluate missing-sequence scenarios.

## Summary

`LongiTumorMamba` combines a robust nnU-Net-style segmentation backbone with modality dropout, spatio-temporal Mamba sequence modeling, previous-mask shape memory, and clinically interpretable tumor-evolution heads. The design is intended to work in both retrospective studies with full longitudinal MRI sequences and prospective clinical follow-up where only the latest scan and prior segmentation are available.
