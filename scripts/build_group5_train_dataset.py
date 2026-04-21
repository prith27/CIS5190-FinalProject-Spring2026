#!/usr/bin/env python3
"""Build Group 5 train subset: shuffle(seed), dedup vs gydou/released_img, export to disk.

Dedup: SHA-256 of RGB image encoded as PNG (lossless, stable per pixel grid).

**Default I/O:** `streaming=True` for the train pool so Hugging Face does **not** build a full
Arrow cache (~many GB). Use ``--materialize`` only if you have enough free disk and want the
classic map-style ``shuffle`` (repro order may differ from streaming shuffle).

See docs/DATA.md Phase B and docs/RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, Value, load_dataset
from PIL import Image


VAL_ID = "gydou/released_img"
TRAIN_POOL_ID = "heidiywseo/5190-image-dataset"


def _png_sha256(pil_image: Image.Image) -> str:
    rgb = pil_image.convert("RGB")
    buf = BytesIO()
    rgb.save(buf, format="PNG", compress_level=6)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _row_image_to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict) and "bytes" in img and img["bytes"]:
        return Image.open(BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, dict) and img.get("path"):
        return Image.open(img["path"]).convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(img)}")


def _collect_val_hashes() -> set[str]:
    val = load_dataset(VAL_ID, split="train", streaming=True)
    hashes: set[str] = set()
    for row in val:
        hashes.add(_png_sha256(_row_image_to_pil(row["image"])))
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deduped train subset for Img2GPS Group 5")
    parser.add_argument("--n", type=int, required=True, help="Target train size after dedup")
    parser.add_argument("--seed", type=int, default=51905, help="Shuffle seed for train pool")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/group5_train"),
        help="Output directory (images/ + metadata.csv + build_manifest.json)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default="",
        metavar="REPO_ID",
        help="If set, push exported dataset to Hugging Face (e.g. org/cis5190-group5-train)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Use with --push-to-hub: create private Hub dataset",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Download+prepare full train split to HF cache (needs lots of disk); map-style shuffle.",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=8192,
        metavar="N",
        help="Streaming shuffle buffer (>= train pool size ~3.4k gives better mix). Default: 8192",
    )
    args = parser.parse_args()

    out: Path = args.out
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading val fingerprints from", VAL_ID, file=sys.stderr)
    val_hashes = _collect_val_hashes()
    print(f"Val unique PNG hashes: {len(val_hashes)}", file=sys.stderr)

    rows_out: list[dict] = []
    skipped_val_dup = 0
    skipped_train_dup = 0
    seen_train: set[str] = set()
    train_io = "materialized" if args.materialize else "streaming"

    if args.materialize:
        print(
            "Materializing train pool (requires large HF cache on disk):",
            TRAIN_POOL_ID,
            file=sys.stderr,
        )
        train = load_dataset(TRAIN_POOL_ID, split="train")
        train = train.shuffle(seed=args.seed)
        for i in range(len(train)):
            row = train[i]
            pil = _row_image_to_pil(row["image"])
            h = _png_sha256(pil)
            if h in val_hashes:
                skipped_val_dup += 1
                continue
            if h in seen_train:
                skipped_train_dup += 1
                continue
            seen_train.add(h)
            rows_out.append(
                {
                    "image": pil,
                    "Latitude": float(row["Latitude"]),
                    "Longitude": float(row["Longitude"]),
                    "_idx": len(rows_out),
                }
            )
            if len(rows_out) >= args.n:
                break
    else:
        print(
            "Streaming train pool (low disk; shuffle buffer=%d):"
            % args.shuffle_buffer_size,
            TRAIN_POOL_ID,
            file=sys.stderr,
        )
        train_stream = load_dataset(TRAIN_POOL_ID, split="train", streaming=True)
        train_stream = train_stream.shuffle(
            seed=args.seed, buffer_size=args.shuffle_buffer_size
        )
        for row in train_stream:
            pil = _row_image_to_pil(row["image"])
            h = _png_sha256(pil)
            if h in val_hashes:
                skipped_val_dup += 1
                continue
            if h in seen_train:
                skipped_train_dup += 1
                continue
            seen_train.add(h)
            rows_out.append(
                {
                    "image": pil,
                    "Latitude": float(row["Latitude"]),
                    "Longitude": float(row["Longitude"]),
                    "_idx": len(rows_out),
                }
            )
            if len(rows_out) >= args.n:
                break

    if len(rows_out) < args.n:
        print(
            f"Warning: only collected {len(rows_out)} rows (requested {args.n}). "
            "Pool may be exhausted after dedup.",
            file=sys.stderr,
        )

    paths: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    meta_lines = ["file_name,Latitude,Longitude"]

    for r in rows_out:
        idx = r["_idx"]
        fname = f"{idx:05d}.jpg"
        fpath = img_dir / fname
        rgb = r["image"].convert("RGB")
        rgb.save(fpath, format="JPEG", quality=92)
        rel = f"images/{fname}"
        paths.append(str(fpath.resolve()))
        lats.append(r["Latitude"])
        lons.append(r["Longitude"])
        meta_lines.append(f"{rel},{r['Latitude']},{r['Longitude']}")

    (out / "metadata.csv").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    manifest = {
        "train_pool": TRAIN_POOL_ID,
        "val_reference": VAL_ID,
        "n_requested": args.n,
        "n_output": len(rows_out),
        "shuffle_seed": args.seed,
        "train_io": train_io,
        "shuffle_buffer_size": (
            args.shuffle_buffer_size if not args.materialize else None
        ),
        "dedup": "sha256_png_rgb",
        "skipped_overlap_with_val": skipped_val_dup,
        "skipped_duplicate_within_train": skipped_train_dup,
        "aug_policy_doc": "docs/DATA.md § Augmentation (Group 5)",
        "aug_policy_id": "group5_randa_m7",
    }
    (out / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(rows_out)} rows to {out}", file=sys.stderr)

    if args.push_to_hub:
        features = Features(
            {
                "image": HFImage(),
                "Latitude": Value("float32"),
                "Longitude": Value("float32"),
            }
        )
        hub_ds = Dataset.from_dict(
            {"image": paths, "Latitude": lats, "Longitude": lons},
            features=features,
        )
        hub_ds.push_to_hub(
            args.push_to_hub,
            private=args.private,
        )
        print("Pushed to", args.push_to_hub, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
