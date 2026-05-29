"""Learning-curve sweep: TRTR macro-F1 as a function of real-training-set size.

For each n_real in {50, 100, 250, 500, 1000, 2000, 2838}, train Twitter-RoBERTa
on a class-stratified random subsample of the real Dreaddit train split, and
evaluate on the full real test split. Three seeds per cell. Outputs the
learning curve we need to choose the "low-resource regime" for follow-up
augmentation experiments.

Run with:
    python code/grid/run_learning_curve.py
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
ROOT = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090")
TRAIN_CSV = ROOT / "CORPORA" / "REDDIT" / "DREADDIT" / "dreaddit-train.csv"
TEST_CSV = ROOT / "CORPORA" / "REDDIT" / "DREADDIT" / "dreaddit-test.csv"
RESULTS = A3 / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-2022-154m"
N_REALS = (50, 100, 250, 500, 1000, 2000, 2838)
SEEDS = (42, 1337, 2024)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--n_reals", type=int, nargs="+", default=list(N_REALS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    rows = []
    out_path = RESULTS / "learning_curve.csv"
    for n in args.n_reals:
        for seed in args.seeds:
            t0 = time.time()
            if n >= len(train_df):
                sub = train_df
            else:
                sub = train_df.groupby("label", group_keys=False).sample(
                    n=n // 2, random_state=seed
                ).reset_index(drop=True)
            tr_texts = sub["text"].tolist()
            tr_labels = sub["label"].tolist()
            macro_f1, acc, per_class = train_eval(
                tr_texts, tr_labels, test_texts, test_labels, seed, device=device,
            )
            elapsed = time.time() - t0
            print(f"n_real={len(sub)} seed={seed} macro_f1={macro_f1:.4f} acc={acc:.4f} time={elapsed:.0f}s")
            rows.append({
                "n_real": len(sub),
                "n_real_requested": n,
                "seed": seed,
                "macro_f1": macro_f1,
                "accuracy": acc,
                "f1_nonstress": per_class[0],
                "f1_stress": per_class[1],
                "seconds": elapsed,
            })
            pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nwrote {out_path}")
    summary = pd.DataFrame(rows).groupby("n_real_requested").agg(
        n_real_mean=("n_real", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS / "learning_curve_summary.csv", index=False)


if __name__ == "__main__":
    main()
