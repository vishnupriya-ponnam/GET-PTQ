"""Qualitative figures: input image, ground-truth mask, FP32 mask and quantized masks."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from .data import IMG_SIZE
import cv2

GT_COLOR, FP32_COLOR, Q_COLOR = (0.0, 0.8, 0.2), (0.2, 0.45, 1.0), (1.0, 0.25, 0.2)


def _input_image(data_dir, name):
    img = np.array(Image.open(os.path.join(data_dir, "images", name + ".png")).convert("RGB"))
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def _outline(mask):
    m = mask.astype(bool)
    return m ^ ndimage.binary_erosion(m, iterations=2)


def _overlay_contours(img, masks_colors):
    out = img.astype(np.float32) / 255.0
    for m, c in masks_colors:
        out[_outline(m)] = c
    return out


def _diff_map(ref, q):
    """White = both foreground, blue = only FP32, red = only quantized, black = both background."""
    out = np.zeros(ref.shape + (3,), np.float32)
    out[(ref == 1) & (q == 1)] = 1.0
    out[(ref == 1) & (q == 0)] = FP32_COLOR
    out[(ref == 0) & (q == 1)] = Q_COLOR
    return out


def pick_examples(per_image: pd.DataFrame, config: str, k: int = 6) -> list[str]:
    """Choose k test images: the largest Dice drops under `config`, typical cases, and the best cases."""
    q = per_image[per_image.config == config].set_index("name")
    f = per_image[per_image.config == "fp32"].set_index("name")
    drop = (f.dice - q.dice).sort_values(ascending=False)
    chosen = list(drop.index[: max(1, k // 3)])                    # most affected by quantization
    by_dice = q.dice.sort_values()
    mid = len(by_dice) // 2
    for n in list(by_dice.index[mid - k:mid + k]) + list(by_dice.index[::-1]):  # typical, then best
        if len(chosen) >= k:
            break
        if n not in chosen:
            chosen.append(n)
    return chosen[:k]


def figure_one_config(preds, per_image, data_dir, config, label, names, path, dataset=""):
    """Rows = test images. Columns: input | ground truth | FP32 | quantized | difference | outlines."""
    idx = {n: i for i, n in enumerate(preds["names"])}
    pi = per_image.set_index(["config", "name"])
    cols = ["Input", "Ground truth", "FP32 mask", f"{label} mask", "FP32 vs quantized", "Outlines"]
    fig, axes = plt.subplots(len(names), len(cols), figsize=(2.3 * len(cols), 2.4 * len(names)), squeeze=False)
    for r, n in enumerate(names):
        i = idx[n]
        img, gt, ref, q = _input_image(data_dir, n), preds["gt"][i], preds["fp32"][i], preds[config][i]
        d_ref, d_q = pi.loc[("fp32", n), "dice"], pi.loc[(config, n), "dice"]
        h_ref, h_q = pi.loc[("fp32", n), "hd95"], pi.loc[(config, n), "hd95"]
        panels = [
            (img, n),
            (gt, "ground truth"),
            (ref, f"Dice {d_ref:.3f} | HD95 {h_ref:.1f}"),
            (q, f"Dice {d_q:.3f} | HD95 {h_q:.1f}"),
            (_diff_map(ref, q), f"{int((ref != q).sum())} pixels differ"),
            (_overlay_contours(img, [(gt, GT_COLOR), (ref, FP32_COLOR), (q, Q_COLOR)]), "green GT, blue FP32, red quant."),
        ]
        for c, (im, sub) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1 if im.ndim == 2 else None)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(sub, fontsize=7)
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
    fig.suptitle(f"{dataset}: ground truth vs. masks before and after quantization ({label})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_all_configs(preds, per_image, data_dir, configs: dict, names, path, dataset=""):
    """Rows = test images. Columns: input | ground truth | one mask per configuration."""
    idx = {n: i for i, n in enumerate(preds["names"])}
    pi = per_image.set_index(["config", "name"])
    slugs = [s for s in configs if s in preds]
    cols = ["Input", "Ground truth"] + [configs[s][0] for s in slugs]
    fig, axes = plt.subplots(len(names), len(cols), figsize=(1.9 * len(cols), 2.1 * len(names)), squeeze=False)
    for r, n in enumerate(names):
        i = idx[n]
        panels = [(_input_image(data_dir, n), n), (preds["gt"][i], "")]
        panels += [(preds[s][i], f"Dice {pi.loc[(s, n), 'dice']:.3f}") for s in slugs]
        for c, (im, sub) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1 if im.ndim == 2 else None)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(sub, fontsize=7)
            if r == 0:
                ax.set_title(cols[c], fontsize=7.5, wrap=True)
    fig.suptitle(f"{dataset}: predicted masks for every precision configuration", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_tradeoff(summary: pd.DataFrame, path, dataset=""):
    """Accuracy vs. speed: one point per TensorRT configuration."""
    s = summary[summary.config != "pytorch_fp32"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, metric, name in [(ax[0], "dice_mean", "Mean Dice"), (ax[1], "hd95_mean", "Mean HD95 (px, lower is better)")]:
        a.scatter(s.latency_total_ms, s[metric], color="#3b6fd8", zorder=3)
        for _, r in s.iterrows():
            a.annotate(r.config, (r.latency_total_ms, r[metric]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        a.set_xlabel("End-to-end latency (ms, batch 1)")
        a.set_ylabel(name)
        a.grid(alpha=0.3)
    fig.suptitle(f"{dataset}: accuracy vs. latency for each precision configuration")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all_figures(out_dir, data_dir, configs: dict, focus_configs=("int8_all",), k: int = 6, dataset=""):
    """Write every figure into out_dir/figures and return their paths."""
    from .experiments import load_predictions

    preds = load_predictions(out_dir)
    per_image = pd.read_csv(os.path.join(out_dir, "per_image.csv"))
    summary = pd.read_csv(os.path.join(out_dir, "summary.csv"))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for cfg in focus_configs:
        if cfg not in preds:
            continue
        names = pick_examples(per_image, cfg, k)
        paths.append(figure_one_config(preds, per_image, data_dir, cfg, configs[cfg][0], names,
                                       os.path.join(fig_dir, f"qualitative_{cfg}.png"), dataset))
    names = pick_examples(per_image, focus_configs[0] if focus_configs[0] in preds else "fp32", k)
    paths.append(figure_all_configs(preds, per_image, data_dir, configs, names,
                                    os.path.join(fig_dir, "qualitative_all_configs.png"), dataset))
    paths.append(figure_tradeoff(summary, os.path.join(fig_dir, "accuracy_vs_latency.png"), dataset))
    return paths


def save_mask_pngs(out_dir, configs=("fp32", "int8_all")):
    """Optionally write every predicted mask as a PNG (out_dir/masks/<config>/<name>.png)."""
    from .experiments import load_predictions

    preds = load_predictions(out_dir)
    for cfg in configs:
        d = os.path.join(out_dir, "masks", cfg)
        os.makedirs(d, exist_ok=True)
        for i, n in enumerate(preds["names"]):
            Image.fromarray(preds[cfg][i] * 255).save(os.path.join(d, f"{n}.png"))
