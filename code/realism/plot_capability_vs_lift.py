"""Plot the decoupling of discriminator AUC from augmentation utility.

For each generator, plot (AUC, lift) at n_real=100 and n_real=250. Uses
letter codes + a legend so labels never overlap. Highlights the Pearson
correlation in each panel — both near zero, demonstrating that realism
does not predict utility.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
SUMMARY = A3 / "results" / "llm_tier_grid_summary.csv"
OUT = A3 / "figures" / "auc_vs_lift.png"

# (cell_key, audit_AUC, display_label, letter_code, family_colour)
GENERATORS = [
    ("Qwen25_300",        0.991, "Qwen 2.5 14B",               "A", "#1f77b4"),
    ("Qwen35_300",        0.996, "Qwen 3.5 9B",                "B", "#1f77b4"),
    ("DeepSeek_300",      0.990, "DeepSeek-R1 14B",            "C", "#1f77b4"),
    ("GemFlash_300",      0.994, "Gemini 2.5 Flash",           "D", "#d62728"),
    ("GemFL_300",         0.994, "Gemini 2.5 Flash-Lite",      "E", "#d62728"),
    ("Gem31FL_300",       0.999, "Gemini 3.1 Flash-Lite",      "F", "#d62728"),
    ("Gemini35Flash_300", 0.999, "Gemini 3.5 Flash",           "G", "#d62728"),
    ("GemFL_FewShot_300", 0.991, "Gemini 2.5 FL few-shot",     "H", "#ff7f0e"),
    ("OASIS_Qwen_300",    0.999, "OASIS + Qwen",               "I", "#9467bd"),
    ("OASIS_Gem_300",     0.998, "OASIS + Gemini",             "J", "#9467bd"),
    ("Claude_50",         0.946, "Claude-authored (n=50)",     "K", "#2ca02c"),
]

FAMILY_LABELS = [
    ("#1f77b4", "Local LLMs (Qwen, DeepSeek)"),
    ("#d62728", "Gemini API"),
    ("#ff7f0e", "Gemini few-shot"),
    ("#9467bd", "OASIS multi-agent"),
    ("#2ca02c", "Human-AI curated"),
]


def _jitter_letters(points, min_dx=0.0005, min_dy=0.008):
    """Tiny x-jitter for points stacked at identical AUC so labels don't overlap."""
    out = []
    by_auc = {}
    for x, y, letter in points:
        by_auc.setdefault(round(x, 4), []).append((x, y, letter))
    for auc, group in by_auc.items():
        if len(group) == 1:
            out.append(group[0])
        else:
            group = sorted(group, key=lambda t: t[1])
            n = len(group)
            for i, (x, y, letter) in enumerate(group):
                dx = (i - (n - 1) / 2) * min_dx
                out.append((x + dx, y, letter))
    return out


def main():
    df = pd.read_csv(SUMMARY)
    base100 = df[(df["cell"] == "TRTR") & (df["n_real"] == 100)]["macro_f1_mean"].iloc[0]
    base250 = df[(df["cell"] == "TRTR") & (df["n_real"] == 250)]["macro_f1_mean"].iloc[0]

    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[3.5, 3.5, 1.6])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2])
    ax_legend.axis("off")

    for panel_idx, (ax, n_real, base) in enumerate(
        zip([ax_a, ax_b], [100, 250], [base100, base250])
    ):
        sub = df[df["n_real"] == n_real]
        points_for_jitter = []
        plotted = []
        for key, auc, label, letter, colour in GENERATORS:
            row = sub[sub["cell"] == key]
            if row.empty:
                continue
            lift = row["macro_f1_mean"].iloc[0] - base
            err = row["macro_f1_std"].iloc[0]
            plotted.append((auc, lift, err, letter, colour))
            points_for_jitter.append((auc, lift, letter))

        # Compute Pearson on un-jittered points (the true values)
        if plotted:
            xs = np.array([p[0] for p in plotted])
            ys = np.array([p[1] for p in plotted])
            r = np.corrcoef(xs, ys)[0, 1]
        else:
            r = float("nan")

        # Jitter same-AUC points for label clarity
        jittered = _jitter_letters(points_for_jitter)
        letter_to_xy = {letter: (x, y) for x, y, letter in jittered}

        for auc, lift, err, letter, colour in plotted:
            x_plot, y_plot = letter_to_xy[letter]
            ax.errorbar(x_plot, lift, yerr=err, fmt="o",
                        color=colour, markersize=11, capsize=3, alpha=0.85,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=3)
            # Letter sits inside the marker
            ax.annotate(letter, (x_plot, lift), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold", zorder=4)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xlabel("Discriminator AUC  (lower = closer to real Dreaddit)", fontsize=10)
        ax.set_ylabel(f"Macro-F1 lift Δ vs TRTR  (n_real = {n_real}, base = {base:.3f})",
                      fontsize=10)
        ax.set_title(f"({chr(97 + panel_idx)})  n_real = {n_real}", fontsize=11)
        ax.grid(True, alpha=0.3)

        ax.text(0.96, 0.96, f"Pearson r = {r:+.3f}\n(n = {len(plotted)} generators)",
                transform=ax.transAxes, fontsize=10, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#666", alpha=0.95))

        ax.set_xlim(0.94, 1.001)

    # Legend panel
    ax_legend.set_title("Generators", fontsize=11, loc="left", fontweight="bold")
    y_top = 0.95
    line_h = 0.045
    family_groups = {
        "#1f77b4": [],
        "#d62728": [],
        "#ff7f0e": [],
        "#9467bd": [],
        "#2ca02c": [],
    }
    for key, auc, label, letter, colour in GENERATORS:
        family_groups[colour].append((letter, label, auc))

    family_titles = {
        "#1f77b4": "Local LLMs",
        "#d62728": "Gemini API (single-shot)",
        "#ff7f0e": "Gemini real-anchored few-shot",
        "#9467bd": "OASIS multi-agent simulation",
        "#2ca02c": "Human-AI curation",
    }
    y = y_top
    for colour, title in family_titles.items():
        members = family_groups[colour]
        if not members:
            continue
        ax_legend.text(0.02, y, title, fontsize=9.5, fontweight="bold", color=colour,
                       transform=ax_legend.transAxes)
        y -= line_h
        for letter, label, auc in members:
            ax_legend.plot([0.04], [y + 0.012], marker="o", color=colour,
                           markersize=11, markeredgecolor="black",
                           markeredgewidth=0.6, transform=ax_legend.transAxes)
            ax_legend.text(0.045, y + 0.013, letter, ha="center", va="center",
                           fontsize=7.5, color="white", fontweight="bold",
                           transform=ax_legend.transAxes)
            ax_legend.text(0.10, y + 0.013, f"{label}  (AUC={auc:.3f})",
                           fontsize=8.5, va="center",
                           transform=ax_legend.transAxes)
            y -= line_h
        y -= 0.015

    fig.suptitle("Discriminator AUC vs augmentation lift — the realism axis does not predict the utility axis",
                 fontsize=12, y=1.02)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
