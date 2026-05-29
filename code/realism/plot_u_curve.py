"""Plot the augmentation U-curve: macro-F1 lift vs n_real for each cell."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
SUMMARY = A3 / "results" / "comprehensive_grid_summary.csv"
OUT = A3 / "figures" / "augmentation_u_curve.png"


CELL_COLOURS = {
    "TRTR": "#000000",
    "EDA_4x": "#7f7f7f",
    "Qwen_2x": "#1f77b4",
    "Qwen_4x": "#d62728",
    "Qwen_max": "#9467bd",
    "QwenFilt1000": "#2ca02c",
    "OASIS_Gem": "#ff7f0e",
}
CELL_LABELS = {
    "TRTR": "TRTR (real-only baseline)",
    "EDA_4x": "EDA × 4 (Wei & Zou 2019)",
    "Qwen_2x": "Qwen 14B × 2",
    "Qwen_4x": "Qwen 14B × 4 (best)",
    "Qwen_max": "Qwen 14B max (all 5,000)",
    "QwenFilt1000": "Qwen + embedding filter (top-1000)",
    "OASIS_Gem": "OASIS multi-agent + Gemini",
}


def main():
    if not SUMMARY.exists():
        raise SystemExit(f"missing {SUMMARY}")
    df = pd.read_csv(SUMMARY)
    base = df[df["cell"] == "TRTR"].set_index("n_real")["macro_f1_mean"]
    if "lift_vs_TRTR" not in df.columns:
        df["lift_vs_TRTR"] = df.apply(
            lambda r: r["macro_f1_mean"] - base.get(r["n_real"], np.nan), axis=1,
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Panel A: macro-F1 vs n_real
    ax = axes[0]
    for cell, sub in df.groupby("cell"):
        sub = sub.sort_values("n_real")
        ax.errorbar(sub["n_real"], sub["macro_f1_mean"], yerr=sub["macro_f1_std"],
                    label=CELL_LABELS.get(cell, cell),
                    color=CELL_COLOURS.get(cell, None),
                    marker="o" if cell != "TRTR" else "s",
                    linewidth=2 if cell == "TRTR" else 1.3,
                    capsize=3, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("n_real (training set size, log scale)")
    ax.set_ylabel("Macro-F1 on Dreaddit held-out test (n=715)")
    ax.set_title("(a) Augmentation cells across the n_real spectrum")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Panel B: lift vs n_real
    ax = axes[1]
    nonbase = df[df["cell"] != "TRTR"]
    for cell, sub in nonbase.groupby("cell"):
        sub = sub.sort_values("n_real")
        ax.plot(sub["n_real"], sub["lift_vs_TRTR"],
                label=CELL_LABELS.get(cell, cell),
                color=CELL_COLOURS.get(cell, None),
                marker="o", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("n_real (training set size, log scale)")
    ax.set_ylabel("Macro-F1 lift vs TRTR")
    ax.set_title("(b) Lift collapses by n_real ≈ 500 (the saturation point)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, alpha=0.3)

    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
