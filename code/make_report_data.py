"""Collate all numbers from the experiment artefacts into a single Markdown snippet.

Reads:
    results/baseline_seeds.csv
    results/baseline_summary.csv
    results/audit_summary.csv
    results/grid_seeds.csv
    results/grid_summary.csv
    results/subgroup_lift.csv
    results/subgroup_mean.csv

Writes:
    results/auto_report_tables.md

The student copies the populated tables into report.md and writes the surrounding prose.
"""

from pathlib import Path
import sys

import pandas as pd

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
R = A3 / "results"


def try_read(name):
    p = R / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def fmt(x, prec=4):
    if pd.isna(x):
        return "—"
    return f"{x:.{prec}f}"


def baseline_block():
    seeds = try_read("baseline_seeds.csv")
    summary = try_read("baseline_summary.csv")
    if seeds is None:
        return "*(baseline_seeds.csv not found)*"
    out = ["**Table 1.** Twitter-RoBERTa fine-tuned on real DREDDIT only, "
           "evaluated on the real held-out test split (n=715).",
           "",
           "| Seed | Macro-F1 | Accuracy | F1 (non-stress) | F1 (stress) |",
           "|------|---------:|---------:|----------------:|------------:|"]
    for _, r in seeds.iterrows():
        out.append(f"| {int(r['seed'])} | {fmt(r['macro_f1'])} | {fmt(r['accuracy'])} | "
                   f"{fmt(r['f1_nonstress'])} | {fmt(r['f1_stress'])} |")
    mean_row = (
        f"| **mean ± std** | "
        f"**{fmt(seeds['macro_f1'].mean())} ± {fmt(seeds['macro_f1'].std())}** | "
        f"{fmt(seeds['accuracy'].mean())} ± {fmt(seeds['accuracy'].std())} | "
        f"{fmt(seeds['f1_nonstress'].mean())} ± {fmt(seeds['f1_nonstress'].std())} | "
        f"{fmt(seeds['f1_stress'].mean())} ± {fmt(seeds['f1_stress'].std())} |"
    )
    out.append(mean_row)
    if summary is not None and "boot_ci_lo" in summary.columns:
        s = summary.iloc[0]
        out.append("")
        out.append(f"Bootstrap 95% CI (2,000 resamples on seed {int(s['boot_ci_basis_seed'])}): "
                   f"[{fmt(s['boot_ci_lo'])}, {fmt(s['boot_ci_hi'])}].")
    return "\n".join(out)


def audit_block():
    a = try_read("audit_summary.csv")
    if a is None:
        return "*(audit_summary.csv not found — run code/realism/audit.py first)*"
    s = a.iloc[0]
    out = ["**Table 2.** Distributional realism audit.",
           "",
           "| Metric | Value |",
           "|--------|------:|",
           f"| Discriminator AUC (5-fold CV, mean ± std) | {fmt(s['discriminator_auc_mean'])} ± {fmt(s['discriminator_auc_std'])} |",
           f"| Vocabulary Jaccard (top-1000 word types) | {fmt(s['vocab_top1000_jaccard'])} |",
           f"| MMD² (real-stress vs synth-stress)      | {fmt(s['mmd2_stress'], 6)} |",
           f"| MMD² (real-non vs synth-non)            | {fmt(s['mmd2_nonstress'], 6)} |"]
    out.append("")
    out.append(f"Sample sizes: n_real = {int(s['n_real'])}, n_synth = {int(s['n_synth'])}.")
    return "\n".join(out)


def grid_block():
    g = try_read("grid_seeds.csv")
    s = try_read("grid_summary.csv")
    if g is None:
        return "*(grid_seeds.csv not found — run code/grid/run_grid.py first)*"
    # one row per config, with n_train shown as the modal training-set size
    by_cfg = g.groupby("config").agg(
        n_train_mode=("n_train", lambda v: int(v.mode().iloc[0])),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        f1_stress_mean=("f1_stress", "mean"),
        f1_nonstress_mean=("f1_nonstress", "mean"),
    ).reset_index()

    if "TRTR" in by_cfg["config"].values:
        baseline_f1 = by_cfg.loc[by_cfg["config"] == "TRTR", "macro_f1_mean"].iloc[0]
    else:
        baseline_f1 = None

    cfg_order = ["TRTR", "TSTR", "Mixed_1x", "Mixed_max", "Real25pct", "Real25pct_Synth"]
    by_cfg["__order"] = by_cfg["config"].map(lambda c: cfg_order.index(c) if c in cfg_order else 99)
    by_cfg = by_cfg.sort_values("__order").drop(columns="__order")

    out = ["**Table 3.** Test macro-F1 by training-set composition. Mean ± std over 5 seeds.",
           "",
           "| Configuration | n_train | Macro-F1 mean ± std | F1 (non-stress) | F1 (stress) | Δ vs TRTR |",
           "|---|---:|---|---:|---:|---:|"]
    for _, r in by_cfg.iterrows():
        delta = "—" if baseline_f1 is None else (
            f"{(r['macro_f1_mean'] - baseline_f1):+.4f}" if r["config"] != "TRTR" else "—"
        )
        out.append(
            f"| {r['config']} | {int(r['n_train_mode']):,} | "
            f"{fmt(r['macro_f1_mean'])} ± {fmt(r['macro_f1_std'])} | "
            f"{fmt(r['f1_nonstress_mean'])} | {fmt(r['f1_stress_mean'])} | {delta} |"
        )
    return "\n".join(out)


def subgroup_block():
    lift = try_read("subgroup_lift.csv")
    mean = try_read("subgroup_mean.csv")
    if lift is None or mean is None:
        return "*(subgroup_*.csv not found — run code/subgroup/run_subgroup.py first)*"
    # The saved CSV has axis + level as the first two cols and one column per config.
    config_cols = [c for c in lift.columns if c not in ("axis", "level")]
    melted = lift.melt(id_vars=["axis", "level"], value_vars=config_cols,
                       var_name="config", value_name="lift")
    melted = melted.dropna(subset=["lift"])
    top_pos = melted.nlargest(5, "lift")
    top_neg = melted.nsmallest(5, "lift")
    out = ["**Subgroup analysis highlights (relative to TRTR)**",
           "",
           "Largest positive lifts:",
           ""]
    for _, r in top_pos.iterrows():
        out.append(f"- `{r['axis']}={r['level']}` under `{r['config']}` → "
                   f"Δmacro-F1 = **{r['lift']:+.4f}**")
    out.append("")
    out.append("Largest negative lifts:")
    out.append("")
    for _, r in top_neg.iterrows():
        out.append(f"- `{r['axis']}={r['level']}` under `{r['config']}` → "
                   f"Δmacro-F1 = **{r['lift']:+.4f}**")
    return "\n".join(out)


def main():
    sections = [
        "# Auto-generated tables for report.md",
        "",
        "*Generated by `python code/make_report_data.py`. Paste the relevant blocks "
        "into `report/report.md` and write the surrounding prose by hand.*",
        "",
        "## 3.1 Baseline",
        "",
        baseline_block(),
        "",
        "## 3.2 Realism audit",
        "",
        audit_block(),
        "",
        "## 3.3 Augmentation grid",
        "",
        grid_block(),
        "",
        "## 3.4 Subgroup analysis",
        "",
        subgroup_block(),
        "",
    ]
    out_path = R / "auto_report_tables.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
