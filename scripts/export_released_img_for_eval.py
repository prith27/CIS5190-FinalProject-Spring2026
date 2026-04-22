#!/usr/bin/env python3
"""Export `gydou/released_img` to a folder layout compatible with `eval_project_a.py`.

Writes `metadata.csv` with columns `file_name`, `Latitude`, `Longitude` and JPEGs under
`images/`, paths relative to the CSV parent directory.

Usage (from repo root):

    python scripts/export_released_img_for_eval.py --out data/val_released_img

See docs/RUNBOOK.md (Phase D).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image


def _row_image_to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict) and img.get("bytes"):
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, dict) and img.get("path"):
        return Image.open(img["path"]).convert("RGB")
    raise TypeError(f"Unsupported image field: {type(img)}")


def main() -> int:
    p = argparse.ArgumentParser(description="Export gydou/released_img for local eval")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/val_released_img"),
        help="Output directory (metadata.csv + images/)",
    )
    p.add_argument(
        "--dataset",
        default="gydou/released_img",
        help="HF dataset id",
    )
    p.add_argument("--split", default="train", help="Split name on Hub (~100 rows)")
    args = p.parse_args()

    out: Path = args.out
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} split={args.split} ...", file=sys.stderr)
    ds = load_dataset(args.dataset, split=args.split)

    rows: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]
        pil = _row_image_to_pil(row["image"])
        fname = f"images/{i:04d}.jpg"
        pil.save(out / fname, format="JPEG", quality=92)
        rows.append(
            {
                "file_name": fname,
                "Latitude": float(row["Latitude"]),
                "Longitude": float(row["Longitude"]),
            }
        )

    csv_path = out / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {len(rows)} rows to {out}", file=sys.stderr)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
