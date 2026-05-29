"""Comprehensive augmentation grid.

For each n_real ∈ {100, 250, 500, 1000, 2838}, train Twitter-RoBERTa on:
  TRTR             real only (control)
  EDA_4x           real + 4× rule-based EDA augmentations of those same real posts
  Qwen_2x          real + 2× LLM (Qwen) synthetic
  Qwen_4x          real + 4× LLM (Qwen) synthetic
  Qwen_max         real + all available Qwen synthetic (typically 5000)
  QwenFilt1000     real + top-1000 most-real-like Qwen synthetic
  OASIS_Gem        real + all OASIS+Gemini multi-agent synthetic (572)

Each cell × 5 seeds, evaluated on the full real Dreaddit held-out test split.

Outputs:
    results/comprehensive_grid_seeds.csv     one row per (n_real, cell, seed)
    results/comprehensive_grid_summary.csv   mean ± std + lift vs TRTR per (n_real, cell)
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
ROOT = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090")
TRAIN_CSV = ROOT / "CORPORA" / "REDDIT" / "DREADDIT" / "dreaddit-train.csv"
TEST_CSV = ROOT / "CORPORA" / "REDDIT" / "DREADDIT" / "dreaddit-test.csv"
RESULTS = A3 / "results"
SYNTH_DIR = A3 / "synth"

sys.path.insert(0, str(A3 / "code" / "synth"))
from eda_augment import eda_one  # noqa: E402

MODEL_NAME = "cardiffnlp/twitter-roberta-base-2022-154m"
SEEDS_DEFAULT = (42, 1337, 2024, 7, 88)
N_REALS_DEFAULT = (100, 250, 500, 1000, 2838)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    if not path or not Path(path).exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def take_synth(records, n, seed):
    if n >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), n, replace=False)
    return [records[i] for i in idx]


def eda_augment_real(real_df, augs_per_post, seed):
    """For each real post, produce `augs_per_post` EDA-augmented variants."""
    import random
    rng = random.Random(seed)
    out_texts, out_labels = [], []
    for _, r in real_df.iterrows():
        for _ in range(augs_per_post):
            text, _ = eda_one(r["text"], alpha=0.1, rng=rng)
            out_texts.append(text)
            out_labels.append(int(r["label"]))
    return out_texts, out_labels


def train_eval(train_texts, train_labels, test_texts, test_labels, seed,
               max_length=128, batch_size=16, lr=2e-5, epochs=3, device="cuda"):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    enc_train = tokenizer(train_texts, truncation=True, padding="max_length",
                          max_length=max_length, return_tensors="pt")
    enc_test = tokenizer(test_texts, truncation=True, padding="max_length",
                         max_length=max_length, return_tensors="pt")
    train_loader = DataLoader(
        TensorDataset(enc_train["input_ids"], enc_train["attention_mask"], torch.tensor(train_labels)),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(enc_test["input_ids"], enc_test["attention_mask"], torch.tensor(test_labels)),
        batch_size=batch_size, shuffle=False,
    )
    optim = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            ids, mask, labels = (t.to(device) for t in batch)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            optim.step()
            optim.zero_grad()
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            ids, mask, _ = (t.to(device) for t in batch)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
    del model
    torch.cuda.empty_cache()
    macro_f1 = f1_score(test_labels, preds, average="macro")
    acc = accuracy_score(test_labels, preds)
    per_class = f1_score(test_labels, preds, average=None)
    return macro_f1, acc, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reals", type=int, nargs="+", default=list(N_REALS_DEFAULT))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--synth_qwen", default=str(SYNTH_DIR / "synth_posts.jsonl"))
    parser.add_argument("--synth_qwen_filt", default=str(SYNTH_DIR / "synth_filtered_top1000.jsonl"))
    parser.add_argument("--synth_oasis_gem", default=str(SYNTH_DIR / "oasis_gemini_60.jsonl"))
    parser.add_argument("--cells", nargs="+",
                        default=["TRTR", "EDA_4x", "Qwen_2x", "Qwen_4x",
                                 "Qwen_max", "QwenFilt1000", "OASIS_Gem"])
    parser.add_argument("--eda_augs", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    real_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    synth_qwen = load_jsonl(args.synth_qwen)
    synth_filt = load_jsonl(args.synth_qwen_filt)
    synth_oasis = load_jsonl(args.synth_oasis_gem)
    print(f"corpora: synth_qwen={len(synth_qwen)} filt1000={len(synth_filt)} "
          f"oasis_gem={len(synth_oasis)}")

    rows = []
    out_path = RESULTS / "comprehensive_grid_seeds.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for n_real in args.n_reals:
        for seed in args.seeds:
            # Stratified real subsample (same posts across all cells at this (n_real, seed))
            if n_real >= len(real_df):
                real_sub = real_df.copy()
            else:
                real_sub = real_df.groupby("label", group_keys=False).sample(
                    n=n_real // 2, random_state=seed
                ).reset_index(drop=True)
            real_texts = real_sub["text"].tolist()
            real_labels = real_sub["label"].tolist()

            for cell in args.cells:
                t0 = time.time()
                if cell == "TRTR":
                    tr_texts, tr_labels = real_texts, real_labels
                elif cell == "EDA_4x":
                    aug_t, aug_l = eda_augment_real(real_sub, args.eda_augs, seed)
                    tr_texts = real_texts + aug_t
                    tr_labels = real_labels + aug_l
                elif cell == "Qwen_2x":
                    extras = take_synth(synth_qwen, 2 * n_real, seed)
                    tr_texts = real_texts + [r["text"] for r in extras]
                    tr_labels = real_labels + [r["label"] for r in extras]
                elif cell == "Qwen_4x":
                    extras = take_synth(synth_qwen, 4 * n_real, seed)
                    tr_texts = real_texts + [r["text"] for r in extras]
                    tr_labels = real_labels + [r["label"] for r in extras]
                elif cell == "Qwen_max":
                    tr_texts = real_texts + [r["text"] for r in synth_qwen]
                    tr_labels = real_labels + [r["label"] for r in synth_qwen]
                elif cell == "QwenFilt1000":
                    tr_texts = real_texts + [r["text"] for r in synth_filt]
                    tr_labels = real_labels + [r["label"] for r in synth_filt]
                elif cell == "OASIS_Gem":
                    tr_texts = real_texts + [r["text"] for r in synth_oasis]
                    tr_labels = real_labels + [r["label"] for r in synth_oasis]
                else:
                    print(f"unknown cell {cell}, skip")
                    continue

                macro_f1, acc, per_class = train_eval(
                    tr_texts, tr_labels, test_texts, test_labels, seed, device=device,
                )
                elapsed = time.time() - t0
                row = {
                    "n_real": n_real,
                    "cell": cell,
                    "seed": seed,
                    "n_train": len(tr_texts),
                    "macro_f1": macro_f1,
                    "accuracy": acc,
                    "f1_nonstress": per_class[0],
                    "f1_stress": per_class[1],
                    "seconds": elapsed,
                }
                rows.append(row)
                print(f"n_real={n_real} cell={cell:12s} seed={seed} "
                      f"n_train={len(tr_texts):5d} macro_f1={macro_f1:.4f} "
                      f"time={elapsed:.0f}s")
                pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby(["n_real", "cell"]).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        n_seeds=("seed", "count"),
        n_train_mean=("n_train", "mean"),
    ).reset_index()
    base = summary[summary["cell"] == "TRTR"].set_index("n_real")["macro_f1_mean"]
    summary["lift_vs_TRTR"] = summary.apply(
        lambda r: r["macro_f1_mean"] - base.get(r["n_real"], np.nan), axis=1,
    )
    summary.to_csv(RESULTS / "comprehensive_grid_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
