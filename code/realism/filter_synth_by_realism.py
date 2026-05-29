"""Embedding-based filtering: keep only the synth posts the discriminator
considers most "real-like".

We train the same logistic-regression discriminator on the full real+synth pool
and then use its probability of "real" for each synthetic post as a real-likeness
score. We keep the top-K synth posts (highest real-likeness) and re-audit. The
filtered subset should have lower discriminator AUC; the question is whether it
also yields better downstream macro-F1 when used as augmentation.

This is a canonical synthetic-data selection technique (cf. importance-weighted
training, "rejection sampling" for synthetic data, Mindermann et al. 2022).

Usage:
    python code/realism/filter_synth_by_realism.py --synth synth/synth_posts.jsonl --top_k 1500
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
ROOT = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090")
TRAIN_CSV = ROOT / "CORPORA" / "REDDIT" / "DREADDIT" / "dreaddit-train.csv"
SYNTH_DIR = A3 / "synth"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def encode(texts, batch_size=64):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return enc.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                      convert_to_numpy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", required=True)
    parser.add_argument("--top_k", type=int, nargs="+", default=[500, 1000, 1500, 2000, 2838])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"loading real {TRAIN_CSV}")
    real_df = pd.read_csv(TRAIN_CSV)
    print(f"loading synth {args.synth}")
    synth_records = load_jsonl(args.synth)
    print(f"real={len(real_df)} synth={len(synth_records)}")

    t0 = time.time()
    real_emb = encode(real_df["text"].tolist())
    synth_texts = [r["text"] for r in synth_records]
    synth_emb = encode(synth_texts)
    print(f"encoding done in {time.time()-t0:.0f}s")

    # Discriminator: predict "real" (y=0) vs "synth" (y=1).
    # We want real-likeness score = P(y=0 | embedding) for each synth post.
    X = np.vstack([real_emb, synth_emb])
    y = np.concatenate([np.zeros(len(real_emb), dtype=int),
                        np.ones(len(synth_emb), dtype=int)])
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0, random_state=args.seed)
    clf.fit(X_s, y)
    real_class_index = list(clf.classes_).index(0)
    synth_probs_real = clf.predict_proba(X_s[len(real_emb):])[:, real_class_index]
    print(f"P(real) on synth — mean={synth_probs_real.mean():.4f}  "
          f"min={synth_probs_real.min():.4f}  max={synth_probs_real.max():.4f}")

    # Sort synth by descending real-likeness; emit ranked JSONL + top-K subsets.
    order = np.argsort(-synth_probs_real)
    ranked_path = SYNTH_DIR / "synth_posts_ranked.jsonl"
    with ranked_path.open("w", encoding="utf-8") as fh:
        for rank, idx in enumerate(order):
            rec = dict(synth_records[idx])
            rec["real_likeness"] = float(synth_probs_real[idx])
            rec["realism_rank"] = rank
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote ranked corpus to {ranked_path}")

    for k in args.top_k:
        keep = order[: min(k, len(order))]
        out = SYNTH_DIR / f"synth_filtered_top{k}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for idx in keep:
                rec = dict(synth_records[idx])
                rec["real_likeness"] = float(synth_probs_real[idx])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Class balance check
        labels = [synth_records[idx]["label"] for idx in keep]
        n1 = sum(labels)
        n0 = len(labels) - n1
        print(f"  top_k={k}: kept {len(keep)} posts | label1={n1} label0={n0} "
              f"(min real-likeness in subset = {synth_probs_real[keep].min():.4f})")


if __name__ == "__main__":
    main()
