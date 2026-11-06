"""Turn summary.csv into a readable table (markdown) for slides and meeting notes."""
from __future__ import annotations

import pandas as pd


def summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compact, rounded table: one row per configuration."""
    s = summary.copy()
    t = pd.DataFrame({
        "Configuration": s["label"],
        "Enc / ETN / Dec": s["encoder"] + " / " + s["etn"] + " / " + s["decoder"],
        "Dice %": (100 * s.dice_mean).map("{:.2f}".format) + " ± " + (100 * s.dice_std).map("{:.2f}".format),
        "IoU %": (100 * s.iou_mean).map("{:.2f}".format),
        "HD95 px": s.hd95_mean.map("{:.2f}".format),
        "ΔDice pp": (100 * s.delta_dice_vs_fp32).map("{:+.2f}".format),
        "ΔHD95 px": s.delta_hd95_vs_fp32.map("{:+.2f}".format),
        "Mask agreement": s.mask_agreement_dice.map("{:.4f}".format),
        "Latency ms": s.latency_total_ms.map("{:.2f}".format),
        "Speedup": s.speedup_vs_fp32.map(lambda v: "" if pd.isna(v) else f"{v:.2f}×"),
        "Engines MB": s.get("engine_mb_total", pd.Series(dtype=float)).map(lambda v: "" if pd.isna(v) else f"{v:.1f}"),
    })
    return t


def summary_markdown(summary: pd.DataFrame, env: dict | None = None) -> str:
    lines = []
    if env:
        whole = "whole test set" if env.get("n_test") == env.get("n_test_full") else \
            f"SUBSET of the {env.get('n_test_full')}-image test set"
        lines.append(f"Dataset **{env.get('dataset')}**, {env.get('n_test')} test images ({whole}), "
                     f"GPU {env.get('gpu')}, TensorRT {env.get('tensorrt', '?')}, "
                     f"{env.get('n_calib')} calibration images, {env.get('calib_method')} calibration.\n")
    t = summary_table(summary)
    try:
        lines.append(t.to_markdown(index=False))
    except ImportError:  # 'tabulate' not installed
        lines.append("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns))
        lines += ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]
    lines.append("\nΔ columns are relative to the FP32 TensorRT engine (negative ΔDice = worse). "
                 "Mask agreement = mean Dice between this configuration's mask and the FP32 mask "
                 "(1.0 = identical). Latency = median end-to-end time per image, batch 1.")
    return "\n".join(lines)
