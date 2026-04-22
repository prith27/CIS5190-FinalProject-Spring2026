"""Submission preprocessing for Project A (Img2GPS).

`prepare_data(csv_path)` reads image metadata and returns `(X, y)` where:
- `X` is a list of model-ready tensors (C, H, W), normalized for ViT input
- `y` is a list of raw `[lat, lon]` labels in decimal degrees

Eval pipeline matches training validation and `training/augmentation.py` policy
`group5_randa_m7` eval branch (`build_eval_transforms`): Resize(256) ->
CenterCrop(224) -> ToTensor -> ImageNet normalization. Training-only augs are
not applied here.

Aligned with run **vit-b16-pf3-v1** (ViT-B/16, partial-freeze 3 blocks); see
`docs/RESULTS.md` and `artifacts/train_log.txt` locally (not committed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image


def _resolve_column(columns: list[str], aliases: list[str]) -> str:
    for name in aliases:
        if name in columns:
            return name
    raise KeyError(f"Could not find columns {aliases} in CSV columns {columns}")


def prepare_data(path: str) -> Tuple[list[torch.Tensor], list[list[float]]]:
    """Load CSV rows and return model inputs + raw degree labels."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()
    file_col = _resolve_column(cols, ["file_name", "filename", "image", "path"])
    lat_col = _resolve_column(cols, ["Latitude", "latitude", "lat"])
    lon_col = _resolve_column(cols, ["Longitude", "longitude", "lon"])

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    x_out: list[torch.Tensor] = []
    y_out: list[list[float]] = []
    parent = csv_path.parent
    for _, row in df.iterrows():
        rel = str(row[file_col])
        image_path = parent / rel
        image = Image.open(image_path).convert("RGB")
        x_out.append(transform(image))
        y_out.append([float(row[lat_col]), float(row[lon_col])])
    return x_out, y_out

