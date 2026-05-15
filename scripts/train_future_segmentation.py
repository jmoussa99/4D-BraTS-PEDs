#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitumor.data import FutureSegmentationDataset, read_manifest
from longitumor.models import LongiTumorMamba, LongiTumorMambaConfig
from longitumor.training import DiceBCELoss
from longitumor.utils import choose_device, parse_patch_size, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train next-visit future tumor segmentation forecasting.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/longitumor_future"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-timepoints", type=int, default=3)
    parser.add_argument("--patch-size", default="96,160,160")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.manifest)
    dataset = FutureSegmentationDataset(
        records,
        input_timepoints=args.input_timepoints,
        patch_size=parse_patch_size(args.patch_size),
    )
    if len(dataset) == 0:
        raise ValueError("No future-training windows found. Need at least input_timepoints + 1 masked visits per patient.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    config = LongiTumorMambaConfig()
    model = LongiTumorMamba(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = DiceBCELoss()
    last_path = args.output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(loader, desc=f"future epoch {epoch}"):
            x = batch["image"].to(device)
            y_future = batch["target"].to(device)
            availability = batch["availability"].to(device)
            delta_t = batch["delta_t"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, availability=availability, delta_t=delta_t)
            loss = criterion(output.logits[:, -1], y_future)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "future_training": {
                "input_timepoints": args.input_timepoints,
                "patch_size": parse_patch_size(args.patch_size),
                "loss": sum(losses) / max(1, len(losses)),
            },
        }
        torch.save(checkpoint, last_path)
        print(f"epoch {epoch} loss {checkpoint['future_training']['loss']:.6f}")

    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
