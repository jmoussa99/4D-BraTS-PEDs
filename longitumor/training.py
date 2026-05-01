from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import (
    LongitudinalMRIDataset,
    SingleVisitMRIDataset,
    collate_single_visit,
    discover_cases,
    read_manifest,
)
from .models import LongiTumorMamba, LongiTumorMambaConfig
from .utils import choose_device


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, probs.ndim))
        intersection = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denom + self.smooth)).mean()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return dice + bce


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-5) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, probs.ndim))
        tp = (probs * target).sum(dim=dims)
        fp = (probs * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - probs) * target).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma).mean()


def temporal_consistency_loss(
    probabilities: torch.Tensor,
    delta_t: torch.Tensor | None = None,
    tau_months: float = 6.0,
) -> torch.Tensor:
    if probabilities.shape[1] < 2:
        return probabilities.new_tensor(0.0)
    diff = (probabilities[:, 1:] - probabilities[:, :-1]).abs().mean(dim=tuple(range(2, probabilities.ndim)))
    if delta_t is None:
        return diff.mean()
    dt = delta_t[:, 1:].to(probabilities.device, probabilities.dtype).clamp_min(0.0)
    weights = torch.exp(-dt / tau_months)
    return (diff * weights).mean()


def shape_memory_loss(probabilities: torch.Tensor, previous_masks: torch.Tensor | None) -> torch.Tensor:
    if previous_masks is None or probabilities.shape[1] < 2:
        return probabilities.new_tensor(0.0)
    return F.l1_loss(probabilities[:, 1:], previous_masks[:, 1:].to(probabilities.device, probabilities.dtype))


def cox_partial_log_likelihood(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order].float()
    log_cumsum_hazard = torch.logcumsumexp(risk, dim=0)
    denom = event.sum().clamp_min(1.0)
    return -((risk - log_cumsum_hazard) * event).sum() / denom


def make_loaders(
    manifest: Path | None,
    data_dir: Path | None,
    patch_size: tuple[int, int, int],
    batch_size: int,
    val_fraction: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    records = read_manifest(manifest) if manifest else discover_cases(data_dir or Path("data"))
    labeled = [record for record in records if record.mask_path]
    if not labeled:
        raise ValueError("No labeled records found for training")
    dataset = SingleVisitMRIDataset(labeled, patch_size=patch_size, training=True)
    val_len = max(1, int(len(dataset) * val_fraction))
    train_len = len(dataset) - val_len
    if train_len < 1:
        raise ValueError("Need at least two labeled cases to create train/validation splits")
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_single_visit)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_single_visit)
    return train_loader, val_loader


def _prepare_single_visit(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> tuple[torch.Tensor, ...]:
    x = batch["image"].to(device).unsqueeze(1)  # type: ignore[union-attr]
    y = batch["target"].to(device).unsqueeze(1)  # type: ignore[union-attr]
    availability = batch["availability"].to(device).unsqueeze(1)  # type: ignore[union-attr]
    delta_t = batch["delta_t"].to(device).unsqueeze(1)  # type: ignore[union-attr]
    return x, y, availability, delta_t


def train_single_timepoint(
    manifest: Path | None = None,
    data_dir: Path | None = Path("data"),
    output_dir: Path = Path("runs/longitumor_single"),
    epochs: int = 20,
    batch_size: int = 1,
    patch_size: tuple[int, int, int] = (96, 160, 160),
    learning_rate: float = 1e-4,
    base_channels: int = 16,
    device_name: str = "auto",
) -> Path:
    device = choose_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(manifest, data_dir, patch_size, batch_size)
    config = LongiTumorMambaConfig(base_channels=base_channels)
    model = LongiTumorMamba(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = DiceBCELoss()
    best_val = float("inf")
    best_path = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train"):
            x, y, availability, delta_t = _prepare_single_visit(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, availability=availability, delta_t=delta_t)
            loss = criterion(output.logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                x, y, availability, delta_t = _prepare_single_visit(batch, device)
                output = model(x, availability=availability, delta_t=delta_t)
                val_losses.append(float(criterion(output.logits, y).cpu()))
        val_loss = sum(val_losses) / max(1, len(val_losses))
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "train_loss": sum(train_losses) / max(1, len(train_losses)),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)
    return best_path


def train_longitudinal(
    manifest: Path,
    output_dir: Path = Path("runs/longitumor_longitudinal"),
    epochs: int = 20,
    batch_size: int = 1,
    patch_size: tuple[int, int, int] = (96, 160, 160),
    learning_rate: float = 5e-5,
    lambda_temp: float = 0.1,
    lambda_shape: float = 0.1,
    device_name: str = "auto",
) -> Path:
    device = choose_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(manifest)
    dataset = LongitudinalMRIDataset(records, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = LongiTumorMamba().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = DiceBCELoss()
    last_path = output_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(loader, desc=f"longitudinal epoch {epoch}"):
            x = batch["image"].to(device)
            y = batch["target"].to(device)
            availability = batch["availability"].to(device)
            delta_t = batch["delta_t"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, availability=availability, delta_t=delta_t)
            loss = criterion(output.logits, y)
            loss = loss + lambda_temp * temporal_consistency_loss(output.probabilities, delta_t)
            loss = loss + lambda_shape * shape_memory_loss(output.probabilities, None)
            loss.backward()
            optimizer.step()
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, last_path)
    return last_path


def run_synthetic_smoke(device_name: str = "auto") -> dict[str, tuple[int, ...] | float]:
    device = choose_device(device_name)
    config = LongiTumorMambaConfig(base_channels=4, embedding_dim=16, use_mamba=False)
    model = LongiTumorMamba(config).to(device)
    model.train()
    x = torch.randn(2, 3, 4, 16, 24, 24, device=device)
    availability = torch.ones(2, 3, 4, device=device)
    target = torch.rand(2, 3, 4, 16, 24, 24, device=device).round()
    previous_masks = torch.zeros_like(target)
    output = model(x, availability=availability, previous_masks=previous_masks)
    loss = DiceBCELoss()(output.logits, target)
    loss.backward()
    return {
        "logits": tuple(output.logits.shape),
        "trajectory_embedding": tuple(output.trajectory_embedding.shape),
        "response_logits": tuple(output.evolution_outputs["response_logits"].shape),
        "risk": tuple(output.evolution_outputs["risk"].shape),
        "loss": float(loss.detach().cpu()),
    }
