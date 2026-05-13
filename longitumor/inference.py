from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

import numpy as np
import torch

from .data import MODALITY_NAMES, VisitRecord, read_manifest, zscore_nonzero
from .models import LongiTumorMamba, LongiTumorMambaConfig
from .utils import choose_device

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    sitk = None
    _sitk_import_error = exc
else:
    _sitk_import_error = None


def _require_sitk() -> None:
    if sitk is None:
        raise ImportError("SimpleITK is required for segmentation mask generation") from _sitk_import_error


def _config_from_checkpoint(checkpoint: dict) -> LongiTumorMambaConfig:
    raw = checkpoint.get("config") or {}
    valid = {field.name for field in fields(LongiTumorMambaConfig)}
    values = {key: value for key, value in raw.items() if key in valid}
    for key in ("channel_multipliers", "modality_dropout"):
        if key in values:
            values[key] = tuple(values[key])
    return LongiTumorMambaConfig(**values)


def load_segmentation_model(checkpoint_path: str | Path, device_name: str = "auto") -> tuple[LongiTumorMamba, torch.device]:
    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = _config_from_checkpoint(checkpoint)
    model = LongiTumorMamba(config).to(device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, device


def _resample_to_reference(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if (
        image.GetSize() == reference.GetSize()
        and image.GetSpacing() == reference.GetSpacing()
        and image.GetOrigin() == reference.GetOrigin()
        and image.GetDirection() == reference.GetDirection()
    ):
        return image
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        image.GetPixelID(),
    )


def _load_modalities(record: VisitRecord) -> tuple[torch.Tensor, torch.Tensor, sitk.Image]:
    _require_sitk()
    reference_path = next((path for path in record.modalities if path), None)
    if reference_path is None:
        raise ValueError(f"No modalities found for {record.patient_id}/{record.visit_id}")

    reference = sitk.ReadImage(str(reference_path))
    reference_shape = tuple(reversed(reference.GetSize()))
    volumes: list[np.ndarray] = []
    availability: list[float] = []
    for path in record.modalities:
        if path:
            image = _resample_to_reference(sitk.ReadImage(str(path)), reference)
            volume = sitk.GetArrayFromImage(image).astype(np.float32)
            volumes.append(zscore_nonzero(volume))
            availability.append(1.0)
        else:
            volumes.append(np.zeros(reference_shape, dtype=np.float32))
            availability.append(0.0)

    image_tensor = torch.from_numpy(np.stack(volumes, axis=0)).unsqueeze(0).unsqueeze(0)
    availability_tensor = torch.tensor(availability, dtype=torch.float32).view(1, 1, len(MODALITY_NAMES))
    return image_tensor, availability_tensor, reference


def _safe_case_id(record: VisitRecord) -> str:
    value = f"{record.patient_id}__{record.visit_id}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", value)


def _write_multimodal_case(record: VisitRecord, input_dir: Path, channel_order: Sequence[str]) -> str:
    _require_sitk()
    modalities = dict(zip(MODALITY_NAMES, record.modalities))
    reference_path = next((modalities.get(name) for name in channel_order if modalities.get(name)), None)
    if reference_path is None:
        raise ValueError(f"No modalities found for {record.patient_id}/{record.visit_id}")

    reference = sitk.ReadImage(str(reference_path))
    case_id = _safe_case_id(record)
    input_dir.mkdir(parents=True, exist_ok=True)
    for channel_index, modality in enumerate(channel_order):
        path = modalities.get(modality)
        if path:
            image = _resample_to_reference(sitk.ReadImage(str(path)), reference)
        else:
            image = sitk.Image(reference.GetSize(), reference.GetPixelID())
            image.CopyInformation(reference)
        sitk.WriteImage(image, str(input_dir / f"{case_id}_{channel_index:04d}.nii.gz"))
    return case_id


def _clear_nifti_outputs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*.nii.gz"):
        item.unlink()


def _resolve_command(command: str, python_executable: str) -> str:
    resolved = shutil.which(command)
    if resolved:
        return resolved
    scripts_dir = Path(python_executable).resolve().parent
    for suffix in ("", ".exe"):
        candidate = scripts_dir / f"{command}{suffix}"
        if candidate.exists():
            return str(candidate)
    return command


def generate_pediatric_brain_tumor_masks(
    records: Sequence[VisitRecord],
    model_repo: str | Path,
    output_dir: str | Path,
    channel_order: Sequence[str] = ("flair", "t1", "t1c", "t2"),
    folds: Sequence[str] = ("0", "1", "2", "3", "4"),
    device: str = "0",
    python_executable: str = sys.executable,
    command: str = "nnUNetv2_predict",
) -> list[Path]:
    """Generate masks with NUBagciLab's pediatric brain tumor model repository.

    The external repository predicts whole tumor and 3-label masks with two
    nnU-Net v2 models, then combines them with postProcessing/conversion.py.
    Its README expects channels ordered as FLAIR, T1, T1C, T2.
    """

    invalid = [name for name in channel_order if name not in MODALITY_NAMES]
    if invalid:
        raise ValueError(f"Unknown channel names: {', '.join(invalid)}")

    repo = Path(model_repo).resolve()
    conversion = repo / "postProcessing" / "conversion.py"
    if not conversion.exists():
        raise FileNotFoundError(f"Could not find pediatric model conversion script: {conversion}")

    post = repo / "postProcessing"
    output_wt = post / "outputWT"
    output_3l = post / "output3L"
    relabeled = post / "relabeled"
    _clear_nifti_outputs(output_wt)
    _clear_nifti_outputs(output_3l)
    _clear_nifti_outputs(relabeled)

    env = os.environ.copy()
    env.setdefault("nnUNet_raw", str(repo / "data" / "nnUNet_raw"))
    env.setdefault("nnUNet_preprocessed", str(repo / "data" / "nnUNet_preprocessed"))
    env.setdefault("nnUNet_results", str(repo / "data" / "nnUNet_results"))
    device_name = device.lower()
    if device_name == "auto":
        nnunet_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_name in {"cpu", "none"}:
        nnunet_device = "cpu"
    else:
        env["CUDA_VISIBLE_DEVICES"] = device
        nnunet_device = "cuda"

    command = _resolve_command(command, python_executable)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="longitumor_pedseg_") as tmp:
        input_dir = Path(tmp) / "images"
        case_ids = [_write_multimodal_case(record, input_dir, channel_order) for record in records]
        base_cmd = [
            command,
            "-i",
            str(input_dir),
            "-f",
            *folds,
            "-tr",
            "nnUNetTrainer",
            "-c",
            "3d_fullres",
            "-p",
            "nnUNetPlans",
            "-device",
            nnunet_device,
        ]
        subprocess.run(
            [*base_cmd, "-d", "Dataset106_WTPED24", "-o", str(output_wt)],
            cwd=repo,
            env=env,
            check=True,
        )
        subprocess.run(
            [*base_cmd, "-d", "Dataset107_3LabelPED24", "-o", str(output_3l)],
            cwd=repo,
            env=env,
            check=True,
        )
        subprocess.run([python_executable, str(conversion)], cwd=repo, env=env, check=True)

        written: list[Path] = []
        for record, case_id in zip(records, case_ids):
            prediction = relabeled / f"{case_id}.nii.gz"
            if not prediction.exists():
                raise FileNotFoundError(f"Pediatric model did not create expected prediction: {prediction}")
            destination = output_root / record.patient_id / f"{record.visit_id}_seg.nii.gz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prediction, destination)
            written.append(destination)
    return written


def probabilities_to_labelmap(probabilities: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    channels = probabilities.detach().cpu().numpy()
    best = channels.argmax(axis=0)
    confidence = channels.max(axis=0)
    label = np.zeros(channels.shape[1:], dtype=np.uint8)
    label[confidence >= threshold] = best[confidence >= threshold].astype(np.uint8) + 1
    return label


def predict_visit_mask(
    model: LongiTumorMamba,
    record: VisitRecord,
    output_path: str | Path,
    device: torch.device,
    threshold: float = 0.5,
) -> Path:
    image, availability, reference = _load_modalities(record)
    image = image.to(device)
    availability = availability.to(device)
    delta_t = torch.tensor([[record.delta_t]], dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(image, availability=availability, delta_t=delta_t)
    label = probabilities_to_labelmap(output.probabilities[0, 0], threshold=threshold)
    mask_image = sitk.GetImageFromArray(label)
    mask_image.CopyInformation(reference)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask_image, str(output))
    return output


def generate_manifest_masks(
    manifest: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    threshold: float = 0.5,
    records: Sequence[VisitRecord] | None = None,
) -> list[Path]:
    model, device = load_segmentation_model(checkpoint, device_name)
    visits = list(records) if records is not None else read_manifest(Path(manifest))
    outputs: list[Path] = []
    root = Path(output_dir)
    for record in visits:
        output = root / record.patient_id / f"{record.visit_id}_seg.nii.gz"
        outputs.append(predict_visit_mask(model, record, output, device, threshold=threshold))
    return outputs
