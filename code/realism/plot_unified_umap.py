"""Unified UMAP showing 3 generators across the realism spectrum.

A single MiniLM + UMAP projection of {real Dreaddit, Qwen 2.5 14B, Gemini 3.5 Flash,
Claude-authored}. Because all four sources share one UMAP coordinate system, the
distributional separability (the discriminator-AUC story of Table 1) becomes
visible: Claude (AUC=0.946) overlaps the real cluster, while Gemini 3.5 Flash
(AUC=0.999, the *newest* and *highest-capability* Gemini Flash tier) sits
furthest from it — visualising the "more polished → more separable → less useful"
hypothesis discussed in §4.

Output: figures/unified_umap.png
"""
from __future__ import annotations
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
ROOT = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090")
TRAIN_CSV = ROOT / "CORPORA" / "REDDIT" / "DREADDIT" / "dreaddit-train.csv"

# (name, jsonl_path, AUC_from_audit, marker_colour, marker_size, max_posts)
SOURCES = [
    ("Real Dreaddit",       None,                                   None,  "#4d7ec8", 6,  None),
    ("Qwen 2.5 14B (local)", A3 / "synth" / "synth_posts.jsonl",     0.991, "#1f77b4", 7, 1500),
    ("Gemini 3.5 Flash (newest API)",
                              A3 / "synth" / "synth_gemini35flash.jsonl",
                                                                     0.999, "#d62728", 9, None),
    ("Claude-authored (hand-curated)",
                              A3 / "synth" / "claude_authored.jsonl", 0.946, "#2ca02c", 11, None),
]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main():
    from sentence_transformers import SentenceTransformer
    import umap

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    rng = np.random.default_rng(42)
    blocks = []  # list of (label, texts)
    real_df = pd.read_csv(TRAIN_CSV)
    blocks.append(("Real Dreaddit", real_df["text"].tolist()))

    for name, path, _, _, _, max_n in SOURCES[1:]:
        rows = load_jsonl(path)
        texts = [r["text"] for r in rows]
        if max_n and len(texts) > max_n:
            idx = rng.choice(len(texts), max_n, replace=False)
            texts = [texts[i] for i in idx]
        blocks.append((name, texts))
        print(f"  {name}: {len(texts)} posts")

    # Encode all blocks with the same MiniLM model
    all_embs = []
    labels = []
    for name, texts in blocks:
        e = enc.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        all_embs.append(e)
        labels.extend([name] * len(texts))
    X = np.vstack(all_embs)
    labels = np.array(labels)
    print(f"  total embeddings: {X.shape}")

    # Single UMAP over all four sources -> shared coordinate system
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.10, metric="cosine", random_state=7)
    proj = reducer.fit_transform(X)
    print(f"  projection shape: {proj.shape}")

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    # Draw real Dreaddit FIRST so synthetic sources sit on top (more readable)
    real_mask = labels == "Real Dreaddit"
    ax.scatter(proj[real_mask, 0], proj[real_mask, 1],
               s=6, alpha=0.25, c="#a9c3e8",
               edgecolors="none", label=None, zorder=1)
    # outline for real cluster
    ax.scatter([], [], s=40, c="#a9c3e8",
               edgecolors="#4d7ec8", linewidths=0.6,
               label=f"Real Dreaddit  (n={int(real_mask.sum()):,})",
               zorder=10)

    for name, path, auc, colour, size, _ in SOURCES[1:]:
        m = labels == name
        n = int(m.sum())
        ax.scatter(proj[m, 0], proj[m, 1], s=size, c=colour, alpha=0.55,
                   edgecolors="black", linewidths=0.25,
                   label=f"{name}  (AUC={auc:.3f}, n={n:,})",
                   zorder=5)

    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title("Shared MiniLM-UMAP projection of real Dreaddit + three synthetic sources\n"
                 "spanning the discriminator-AUC spectrum  (lower AUC = closer to real)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
              title="Source  (AUC vs Dreaddit-train)", title_fontsize=9.5)
    ax.grid(True, alpha=0.25)

    out = A3 / "figures" / "unified_umap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
