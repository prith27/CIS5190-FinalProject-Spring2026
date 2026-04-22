#!/usr/bin/env python3
"""Train ViT-B/16 regression for Img2GPS with partial freezing.

This script trains on the Group 5 train dataset and validates on `gydou/released_img`.
It is designed for EC2 runs with crash-safe checkpointing and resume support.

Key design points:
- ViT-B/16 backbone with 2D regression head (`lat`, `lon` in normalized space)
- Partial freeze (default: unfreeze last 3 encoder blocks + encoder.ln + head)
- Differential learning rates for head vs unfrozen backbone
- AMP support on CUDA
- Checkpoints every epoch (`latest_checkpoint.pt`) and best-by-val-Haversine
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Running `python training/run_train_vit.py` puts `training/` on sys.path, not the repo root.
# Ensure project root is importable so `training.augmentation` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.models import ViT_B_16_Weights, vit_b_16

from training.augmentation import AUG_POLICY_ID, build_eval_transforms, build_train_transforms


EARTH_RADIUS_M = 6_371_000.0


def parse_args() -> argparse.Namespace:
    """Parse CLI args for EC2/local training runs."""
    p = argparse.ArgumentParser(description="Train ViT-B/16 regression for Img2GPS")
    p.add_argument("--hf-train", default="prith27/cis5190-group5-train", help="HF dataset id for train split")
    p.add_argument("--hf-val", default="gydou/released_img", help="HF dataset id for validation split")
    p.add_argument("--train-split", default="train", help="Split name for train dataset")
    p.add_argument("--val-split", default="train", help="Split name for val dataset")
    p.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for regression head")
    p.add_argument("--backbone-lr", type=float, default=1e-5, help="Learning rate for unfrozen backbone blocks")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    p.add_argument("--unfreeze-blocks", type=int, default=3, help="How many trailing encoder blocks to unfreeze")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=51905, help="Global random seed")
    p.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    p.add_argument("--output-dir", type=str, default="artifacts", help="Directory for logs/checkpoints")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_column(columns: list[str], aliases: list[str]) -> str:
    for name in aliases:
        if name in columns:
            return name
    raise KeyError(f"Could not find columns {aliases} in dataset columns {columns}")


def _to_pil(image_field: Any) -> Image.Image:
    """Convert HF `image` field to PIL RGB."""
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict):
        if image_field.get("path"):
            return Image.open(image_field["path"]).convert("RGB")
        if image_field.get("bytes"):
            import io

            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(image_field)}")


@dataclass(frozen=True)
class NormStats:
    lat_mean: float
    lat_std: float
    lon_mean: float
    lon_std: float

    def to_dict(self) -> dict[str, float]:
        return {
            "lat_mean": self.lat_mean,
            "lat_std": self.lat_std,
            "lon_mean": self.lon_mean,
            "lon_std": self.lon_std,
        }


def compute_norm_stats(ds: Dataset, lat_col: str, lon_col: str) -> NormStats:
    """Compute z-score stats on train labels."""
    lats = np.asarray([float(v) for v in ds[lat_col]], dtype=np.float64)
    lons = np.asarray([float(v) for v in ds[lon_col]], dtype=np.float64)
    lat_std = float(lats.std())
    lon_std = float(lons.std())
    if lat_std <= 0.0 or lon_std <= 0.0:
        raise ValueError("Lat/Lon std must be > 0 to normalize targets.")
    return NormStats(
        lat_mean=float(lats.mean()),
        lat_std=lat_std,
        lon_mean=float(lons.mean()),
        lon_std=lon_std,
    )


class HFRegressionDataset(TorchDataset):
    """HF Dataset wrapper returning transformed image tensor + normalized target."""

    def __init__(
        self,
        hf_ds: Dataset,
        transform,
        lat_col: str,
        lon_col: str,
        norm: NormStats,
    ) -> None:
        self.hf_ds = hf_ds
        self.transform = transform
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.norm = norm

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.hf_ds[idx]
        image = _to_pil(row["image"])
        x = self.transform(image)
        lat = float(row[self.lat_col])
        lon = float(row[self.lon_col])
        target_raw = torch.tensor([lat, lon], dtype=torch.float32)
        target_norm = torch.tensor(
            [
                (lat - self.norm.lat_mean) / self.norm.lat_std,
                (lon - self.norm.lon_mean) / self.norm.lon_std,
            ],
            dtype=torch.float32,
        )
        return x, target_norm, target_raw


def haversine_m_batch(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance in meters."""
    lat1 = np.radians(pred_deg[:, 0])
    lon1 = np.radians(pred_deg[:, 1])
    lat2 = np.radians(true_deg[:, 0])
    lon2 = np.radians(true_deg[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))


def denorm_outputs(outputs: torch.Tensor, norm: NormStats) -> torch.Tensor:
    """Convert normalized model outputs back to raw degree space."""
    lat = outputs[:, 0] * norm.lat_std + norm.lat_mean
    lon = outputs[:, 1] * norm.lon_std + norm.lon_mean
    return torch.stack([lat, lon], dim=1)


def freeze_for_partial_ft(model: nn.Module, unfreeze_blocks: int) -> tuple[int, int]:
    """Freeze all params, then unfreeze trailing encoder blocks + ln + head."""
    for p in model.parameters():
        p.requires_grad = False

    if unfreeze_blocks < 0 or unfreeze_blocks > 12:
        raise ValueError("--unfreeze-blocks must be in [0, 12] for ViT-B/16")

    if unfreeze_blocks > 0:
        for block in model.encoder.layers[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    for p in model.encoder.ln.parameters():
        p.requires_grad = True
    for p in model.heads.head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm: NormStats,
    criterion: nn.Module,
) -> dict[str, float]:
    """Run validation and return MSE loss + raw-degree metrics."""
    model.eval()
    losses: list[float] = []
    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    with torch.no_grad():
        for x, target_norm, target_raw in loader:
            x = x.to(device, non_blocking=True)
            target_norm = target_norm.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, target_norm)
            losses.append(float(loss.item()))
            pred_raw = denorm_outputs(out, norm).cpu().numpy()
            pred_batches.append(pred_raw)
            true_batches.append(target_raw.numpy())

    preds = np.concatenate(pred_batches, axis=0)
    trues = np.concatenate(true_batches, axis=0)
    distances = haversine_m_batch(preds, trues)
    mae = np.abs(preds - trues).mean()
    rmse = math.sqrt(np.square(preds - trues).mean())
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_haversine_m": float(distances.mean()) if distances.size else float("nan"),
        "val_mae_deg": float(mae),
        "val_rmse_deg": float(rmse),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save checkpoint to avoid partial writes."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = out_dir / "latest_checkpoint.pt"
    best_ckpt = out_dir / "best_model.pt"
    final_model_path = out_dir / "model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    print(f"device={device} amp={use_amp} aug_policy={AUG_POLICY_ID}")
    print(f"loading datasets: train={args.hf_train} val={args.hf_val}")

    train_hf = load_dataset(args.hf_train, split=args.train_split)
    val_hf = load_dataset(args.hf_val, split=args.val_split)

    columns = list(train_hf.column_names)
    lat_col = _resolve_column(columns, ["Latitude", "latitude", "lat"])
    lon_col = _resolve_column(columns, ["Longitude", "longitude", "lon"])

    norm = compute_norm_stats(train_hf, lat_col=lat_col, lon_col=lon_col)
    print("Normalization stats (copy into model.py):")
    print(f"LAT_MEAN={norm.lat_mean:.10f}")
    print(f"LAT_STD={norm.lat_std:.10f}")
    print(f"LON_MEAN={norm.lon_mean:.10f}")
    print(f"LON_STD={norm.lon_std:.10f}")

    (out_dir / "norm_stats.json").write_text(json.dumps(norm.to_dict(), indent=2) + "\n", encoding="utf-8")

    train_ds = HFRegressionDataset(
        hf_ds=train_hf,
        transform=build_train_transforms(),
        lat_col=lat_col,
        lon_col=lon_col,
        norm=norm,
    )
    val_ds = HFRegressionDataset(
        hf_ds=val_hf,
        transform=build_eval_transforms(),
        lat_col=lat_col,
        lon_col=lon_col,
        norm=norm,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(768, 2)
    trainable, total = freeze_for_partial_ft(model, args.unfreeze_blocks)
    model = model.to(device)

    print(
        f"unfreeze_blocks={args.unfreeze_blocks} "
        f"trainable_params={trainable} total_params={total}"
    )

    head_params = list(model.heads.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("heads.head")]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_val_h = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_h = float(ckpt.get("best_val_haversine", float("inf")))
        print(f"resumed_from={resume_path} start_epoch={start_epoch} best_val_h={best_val_h:.3f}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        steps = 0
        for x, target_norm, _target_raw in train_loader:
            x = x.to(device, non_blocking=True)
            target_norm = target_norm.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp) if device.type == "cuda" else contextlib.nullcontext()
            with autocast_ctx:
                out = model(x)
                loss = criterion(out, target_norm)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            steps += 1

        scheduler.step()
        train_loss = running_loss / max(steps, 1)

        metrics = {
            "val_loss": float("nan"),
            "val_haversine_m": float("nan"),
            "val_mae_deg": float("nan"),
            "val_rmse_deg": float("nan"),
        }
        if (epoch + 1) % max(args.val_every, 1) == 0:
            metrics = evaluate(model, val_loader, device, norm, criterion)

        epoch_sec = time.time() - t0
        lr_head = optimizer.param_groups[0]["lr"]
        lr_backbone = optimizer.param_groups[1]["lr"]
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_haversine_m={metrics['val_haversine_m']:.3f} "
            f"val_mae_deg={metrics['val_mae_deg']:.6f} "
            f"val_rmse_deg={metrics['val_rmse_deg']:.6f} "
            f"lr_head={lr_head:.2e} lr_backbone={lr_backbone:.2e} "
            f"time_s={epoch_sec:.1f}"
        )

        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "best_val_haversine": best_val_h,
            "val_metrics": metrics,
            "norm_stats": norm.to_dict(),
            "unfreeze_blocks": args.unfreeze_blocks,
            "config": vars(args),
        }
        save_checkpoint(latest_ckpt, payload)

        val_h = metrics["val_haversine_m"]
        if not math.isnan(val_h) and val_h < best_val_h:
            best_val_h = val_h
            payload["best_val_haversine"] = best_val_h
            save_checkpoint(best_ckpt, payload)
            print(f"best_checkpoint_updated val_haversine_m={best_val_h:.3f}")

    if best_ckpt.exists():
        best_payload = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(best_payload["model_state_dict"])
        print(f"loaded_best_checkpoint val_haversine_m={best_payload.get('best_val_haversine', float('nan')):.3f}")
    else:
        print("warning: no best checkpoint found; exporting current model weights")

    torch.save(model.state_dict(), final_model_path)
    print(f"saved_final_state_dict={final_model_path}")
    print("done")


if __name__ == "__main__":
    main()

