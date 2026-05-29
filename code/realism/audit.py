"""Stage 3: distributional realism audit.

For each of the four standard tests below we report a single, defensible number:

1. UMAP overlay   — qualitative figure: real vs synthetic in 2D embedding space.
2. Discriminator AUC — train a logistic regression to predict real vs synth from
   sentence embeddings. AUC near 0.5 means indistinguishable, near 1.0 means
   trivially separable. Reported with 5-fold CV.
3. Vocabulary Jaccard — top-1000 word types (after simple lowercasing/regex), real
   vs synth, Jaccard intersection-over-union.
4. Class-conditional MMD — squared maximum-mean-discrepancy with RBF kernel
   between {real-stress vs synth-stress} and {real-nonstress vs synth-nonstress}.
   Lower = more similar distribution. Reported with a bootstrap on real subsamples.

Run with:
    python code/realism/audit.py --synth synth/synth_posts.jsonl
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
ROOT = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090")
TRAIN_CSV = ROOT / "CORPORA" / "REDDIT" / "DREADDIT" / "dreaddit-train.csv"

RESULTS = A3 / "results"
FIGS = A3 / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


WORD_RE = re.compile(r"[a-zA-Z']{2,}")


def load_synth(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def encode_with_minilm(texts, batch_size=64):
    """Sentence embeddings via all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return enc.encode(list(texts), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)


# --- 1. UMAP figure ---------------------------------------------------------


def plot_umap(real_emb, synth_emb, real_labels, synth_labels, out_path):
    import umap
    import matplotlib.pyplot as plt

    rng_state = 7
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, metric="cosine", random_state=rng_state)
    all_emb = np.vstack([real_emb, synth_emb])
    proj = reducer.fit_transform(all_emb)
    proj_real = proj[: len(real_emb)]
    proj_synth = proj[len(real_emb) :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].scatter(proj_synth[:, 0], proj_synth[:, 1], s=4, alpha=0.30,
                    c="#d62728", label="synthetic")
    axes[0].scatter(proj_real[:, 0], proj_real[:, 1], s=4, alpha=0.45,
                    c="#1f77b4", label="DREDDIT")
    axes[0].set_title("Real vs synthetic posts (UMAP)")
    axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")
    axes[0].legend(loc="lower left", markerscale=2)

    real_s = proj_real[np.asarray(real_labels) == 1]
    real_n = proj_real[np.asarray(real_labels) == 0]
    synth_s = proj_synth[np.asarray(synth_labels) == 1]
    synth_n = proj_synth[np.asarray(synth_labels) == 0]
    axes[1].scatter(synth_n[:, 0], synth_n[:, 1], s=4, alpha=0.30, c="#2ca02c", label="synth non-stress")
    axes[1].scatter(synth_s[:, 0], synth_s[:, 1], s=4, alpha=0.30, c="#d62728", label="synth stress")
    axes[1].scatter(real_n[:, 0], real_n[:, 1], s=4, alpha=0.45, c="#1f77b4", label="real non-stress")
    axes[1].scatter(real_s[:, 0], real_s[:, 1], s=4, alpha=0.45, c="#9467bd", label="real stress")
    axes[1].set_title("Stress vs non-stress by source")
    axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")
    axes[1].legend(loc="lower left", markerscale=2, fontsize=9)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --- 2. Discriminator AUC ----------------------------------------------------


def discriminator_auc(real_emb, synth_emb, n_splits=5, seed=42):
    X = np.vstack([real_emb, synth_emb])
    y = np.concatenate([np.zeros(len(real_emb), dtype=int), np.ones(len(synth_emb), dtype=int)])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0)
        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], probs))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


# --- 3. Vocabulary Jaccard ---------------------------------------------------


def top_words(texts, top_k=1000):
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(w.lower() for w in WORD_RE.findall(t))
    return set(w for w, _ in cnt.most_common(top_k))


def vocab_jaccard(real_texts, synth_texts, top_k=1000):
    r = top_words(real_texts, top_k)
    s = top_words(synth_texts, top_k)
    if not r or not s:
        return 0.0
    return len(r & s) / len(r | s)


# --- 4. Class-conditional MMD ------------------------------------------------


def rbf_mmd2(X, Y, sigma):
    """Unbiased estimator of MMD^2 with RBF kernel (Gretton et al. 2012)."""
    XX = np.exp(-((X[:, None] - X[None, :]) ** 2).sum(-1) / (2 * sigma * sigma))
    YY = np.exp(-((Y[:, None] - Y[None, :]) ** 2).sum(-1) / (2 * sigma * sigma))
    XY = np.exp(-((X[:, None] - Y[None, :]) ** 2).sum(-1) / (2 * sigma * sigma))
    m, n = len(X), len(Y)
    sum_xx = (XX.sum() - np.trace(XX)) / (m * (m - 1))
    sum_yy = (YY.sum() - np.trace(YY)) / (n * (n - 1))
    sum_xy = XY.sum() / (m * n)
    return float(sum_xx + sum_yy - 2 * sum_xy)


def median_heuristic_sigma(X, Y, cap=2000, seed=0):
    rng = np.random.default_rng(seed)
    sample = np.vstack([X, Y])
    if len(sample) > cap:
        idx = rng.choice(len(sample), cap, replace=False)
        sample = sample[idx]
    d = np.linalg.norm(sample[:, None] - sample[None, :], axis=-1)
    triu = d[np.triu_indices_from(d, k=1)]
    return float(np.median(triu)) or 1.0


def class_conditional_mmd(real_emb, real_y, synth_emb, synth_y, n_boot=10, cap=400, seed=42):
    rng = np.random.default_rng(seed)
    results = {}
    for cls in (0, 1):
        Xr = real_emb[np.asarray(real_y) == cls]
        Xs = synth_emb[np.asarray(synth_y) == cls]
        if len(Xr) < 10 or len(Xs) < 10:
            results[cls] = float("nan")
            continue
        boots = []
        sigma = median_heuristic_sigma(Xr, Xs)
        for _ in range(n_boot):
            ri = rng.choice(len(Xr), min(cap, len(Xr)), replace=False)
            si = rng.choice(len(Xs), min(cap, len(Xs)), replace=False)
            boots.append(rbf_mmd2(Xr[ri], Xs[si], sigma))
        results[cls] = float(np.mean(boots))
    return results


# --- driver ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", default=str(A3 / "synth" / "synth_posts.jsonl"))
    parser.add_argument("--sample_real", type=int, default=0, help="0 = use all real train posts")
    parser.add_argument("--sample_synth", type=int, default=0, help="0 = use all synth posts")
    parser.add_argument("--tag", default="audit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"loading real from {TRAIN_CSV}")
    real = pd.read_csv(TRAIN_CSV)
    print(f"loading synth from {args.synth}")
    synth = load_synth(args.synth)
    print(f"real={len(real)} synth={len(synth)}")

    rng = np.random.default_rng(args.seed)
    if args.sample_real and args.sample_real < len(real):
        real = real.sample(args.sample_real, random_state=args.seed).reset_index(drop=True)
    if args.sample_synth and args.sample_synth < len(synth):
        synth = synth.sample(args.sample_synth, random_state=args.seed).reset_index(drop=True)

    t0 = time.time()
    real_emb = encode_with_minilm(real["text"].tolist())
    synth_emb = encode_with_minilm(synth["text"].tolist())
    print(f"encoding done in {time.time()-t0:.0f}s -- real_emb={real_emb.shape} synth_emb={synth_emb.shape}")

    fig_path = FIGS / f"{args.tag}_umap.png"
    print("rendering UMAP figure ...")
    plot_umap(real_emb, synth_emb, real["label"].tolist(), synth["label"].tolist(), fig_path)
    print(f"saved {fig_path}")

    print("running discriminator ...")
    auc_mean, auc_std, fold_aucs = discriminator_auc(real_emb, synth_emb, seed=args.seed)
    print(f"discriminator_auc mean={auc_mean:.4f} std={auc_std:.4f} folds={fold_aucs}")

    print("running vocab Jaccard ...")
    jac = vocab_jaccard(real["text"], synth["text"])
    print(f"vocab_top1000_jaccard={jac:.4f}")

    print("running class-conditional MMD ...")
    mmd = class_conditional_mmd(real_emb, real["label"], synth_emb, synth["label"], seed=args.seed)
    print(f"mmd_nonstress={mmd[0]:.6f} mmd_stress={mmd[1]:.6f}")

    summary = {
        "tag": args.tag,
        "n_real": int(len(real)),
        "n_synth": int(len(synth)),
        "discriminator_auc_mean": auc_mean,
        "discriminator_auc_std": auc_std,
        "vocab_top1000_jaccard": jac,
        "mmd2_nonstress": mmd[0],
        "mmd2_stress": mmd[1],
    }
    out_csv = RESULTS / f"{args.tag}_summary.csv"
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
