"""Submission model for Project A (Img2GPS).

Contract:
- Exposes `Model` and `get_model()`
- `Model()` is instantiable with no required arguments
- `predict(batch)` returns a list of `[lat, lon]` pairs in raw decimal degrees

Train-set z-score stats (must match `training/run_train_vit.py` / `artifacts/norm_stats.json`
for the submitted `model.pt`). Source: `prith27/cis5190-group5-train`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import torch
from torch import nn
from torchvision.models import vit_b_16

# From artifacts/norm_stats.json (run vit-b16-pf3-v1, 2026-04-22).
LAT_MEAN = 39.951561176300046
LAT_STD = 0.0006491282736708259
LON_MEAN = -75.19154781341553
LON_STD = 0.0005868093627598868


class Model(nn.Module):
    """ViT-B/16 regressor wrapper used by the evaluator."""

    def __init__(self, weights_path: str | None = None) -> None:
        """Build architecture; optionally load checkpoint if a valid path is provided."""
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = vit_b_16(weights=None)
        self.model.heads.head = nn.Linear(768, 2)
        self.model.to(self.device)
        self.model.eval()

        self.lat_mean = float(LAT_MEAN)
        self.lat_std = float(LAT_STD)
        self.lon_mean = float(LON_MEAN)
        self.lon_std = float(LON_STD)

        if weights_path:
            path = Path(weights_path)
            if path.exists():
                state = torch.load(path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if not isinstance(state, dict):
                    raise RuntimeError("weights_path must point to a state_dict checkpoint")
                self.load_state_dict(state, strict=False)
                self.model.eval()

    def eval(self) -> "Model":
        """Set module to evaluation mode and return self."""
        super().eval()
        self.model.eval()
        return self

    def _to_tensor_batch(self, batch: Iterable[Any]) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                t = item
            else:
                arr = np.asarray(item, dtype=np.float32)
                t = torch.from_numpy(arr)
            if t.ndim != 3:
                raise ValueError(f"Expected each input item shape (C,H,W), got {tuple(t.shape)}")
            tensors.append(t)
        if not tensors:
            return torch.empty((0, 3, 224, 224), dtype=torch.float32)
        return torch.stack(tensors, dim=0).to(self.device, dtype=torch.float32)

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """Predict raw `[lat, lon]` degree pairs for the input batch."""
        x = self._to_tensor_batch(batch)
        if x.shape[0] == 0:
            return []
        with torch.no_grad():
            out = self.model(x)
            lat = out[:, 0] * self.lat_std + self.lat_mean
            lon = out[:, 1] * self.lon_std + self.lon_mean
            pred = torch.stack([lat, lon], dim=1).cpu().numpy()
        return pred.tolist()


def get_model() -> Model:
    """Factory required by the evaluator."""
    return Model()

