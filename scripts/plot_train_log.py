"""
Parse artifacts/train_log.txt (ViT trainer epoch lines) and write matplotlib PNGs
under artifacts/plots/. Run from repo root:

  python scripts/plot_train_log.py
  python scripts/plot_train_log.py --log artifacts/train_log.txt --out artifacts/plots
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# epoch=15/20 train_loss=0.126394 val_haversine_m=62.176 val_mae_deg=0.000404 ...
LINE_RE = re.compile(
    r"epoch=(?P<e>\d+)/(?P<e_tot>\d+)\s+"
    r"train_loss=(?P<tr>[\d.]+)\s+"
    r"val_haversine_m=(?P<vh>[\d.]+)\s+"
    r"val_mae_deg=(?P<mae>[\d.]+)\s+"
    r"val_rmse_deg=(?P<rmse>[\d.]+)\s+"
    r"lr_head=(?P<lrh>[^\s]+)\s+lr_backbone=(?P<lrb>[^\s]+)\s+"
    r"time_s=(?P<t>[\d.]+)"
)


def parse_epoch_lines(text: str) -> dict[str, list]:
    epochs: list[int] = []
    train_loss: list[float] = []
    val_haversine_m: list[float] = []
    val_mae_deg: list[float] = []
    val_rmse_deg: list[float] = []
    lr_head: list[float] = []
    lr_backbone: list[float] = []
    epoch_time_s: list[float] = []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("epoch="):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        epochs.append(int(m.group("e")))
        train_loss.append(float(m.group("tr")))
        val_haversine_m.append(float(m.group("vh")))
        val_mae_deg.append(float(m.group("mae")))
        val_rmse_deg.append(float(m.group("rmse")))
        lr_head.append(float(m.group("lrh")))
        lr_backbone.append(float(m.group("lrb")))
        epoch_time_s.append(float(m.group("t")))

    return {
        "epoch": epochs,
        "train_loss": train_loss,
        "val_haversine_m": val_haversine_m,
        "val_mae_deg": val_mae_deg,
        "val_rmse_deg": val_rmse_deg,
        "lr_head": lr_head,
        "lr_backbone": lr_backbone,
        "epoch_time_s": epoch_time_s,
    }


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_xmargin(0.02)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot metrics from ViT train_log.txt")
    p.add_argument("--log", type=Path, default=Path("artifacts/train_log.txt"))
    p.add_argument("--out", type=Path, default=Path("artifacts/plots"))
    args = p.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    d = parse_epoch_lines(text)
    ep = d["epoch"]
    if not ep:
        raise SystemExit(
            f"No epoch=... lines matched in {args.log}. Expected format from training/run_train_vit.py."
        )

    meta_line = ""
    for line in text.splitlines():
        if line.startswith("loading datasets:"):
            meta_line = line.strip()
            break

    best_i = min(range(len(d["val_haversine_m"])), key=lambda i: d["val_haversine_m"][i])
    best_ep = ep[best_i]
    best_h = d["val_haversine_m"][best_i]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, d["train_loss"], marker="o", markersize=4, label="Train loss (MSE, normalized targets)")
    style_axes(ax, "Training loss vs epoch", "Epoch", "Loss")
    if meta_line:
        ax.text(0.02, 0.98, meta_line[:120], transform=ax.transAxes, fontsize=7, verticalalignment="top", alpha=0.75)
    ax.legend(loc="upper right")
    savefig(args.out / "train_loss.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, d["val_haversine_m"], marker="o", markersize=4, color="#c44e52", label="Val mean Haversine (m)")
    ax.axhline(best_h, color="gray", linestyle=":", alpha=0.8, linewidth=1)
    ax.scatter([best_ep], [best_h], s=120, zorder=5, color="#000000", edgecolors="white", linewidths=1)
    ax.annotate(f"best ep {best_ep}\n{best_h:.2f} m", xy=(best_ep, best_h), xytext=(5, 12), textcoords="offset points")
    style_axes(ax, "Validation localization error vs epoch", "Epoch", "Mean Haversine distance (m)")
    ax.legend(loc="upper right")
    subtitle = "(Trainer does not log val MSE separately; Haversine on val is the main val metric.)"
    fig.text(0.5, 0.02, subtitle, ha="center", fontsize=8, style="italic", color="dimgray")
    savefig(args.out / "val_haversine_m.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, d["val_mae_deg"], marker="s", markersize=3, label="Val MAE (deg)")
    ax.plot(ep, d["val_rmse_deg"], marker="^", markersize=3, label="Val RMSE (deg)")
    style_axes(ax, "Validation coordinate error vs epoch", "Epoch", "Error (degrees)")
    ax.legend(loc="upper right")
    savefig(args.out / "val_deg_errors.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, d["lr_head"], label="LR head")
    ax.plot(ep, d["lr_backbone"], label="LR backbone")
    ax.set_yscale("log")
    style_axes(ax, "Learning rates (AdamW groups)", "Epoch", "Learning rate (log scale)")
    ax.legend(loc="upper right")
    savefig(args.out / "learning_rates.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ep, d["train_loss"], marker="o", markersize=3)
    style_axes(ax1, "Train loss", "Epoch", "MSE train loss")
    ax2.plot(ep, d["val_haversine_m"], marker="o", markersize=3, color="#c44e52")
    style_axes(ax2, "Val Haversine (m)", "Epoch", "meters")
    fig.suptitle("Img2GPS ViT training summary", fontsize=12)
    savefig(args.out / "overview_train_val.png")


if __name__ == "__main__":
    main()
