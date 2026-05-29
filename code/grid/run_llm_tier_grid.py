"""Phase 2 grid: LLM-tier × n_real comparison at matched synth budget.

Tests the hypothesis: does a *more capable* LLM (Gemini 3.1 Flash-Lite) give
a larger augmentation lift than a less capable one (Qwen 14B) at the same N?

For each n_real ∈ {100, 250} and seed ∈ {42, 1337, 2024, 7, 88}, train
Twitter-RoBERTa on:

  TRTR              real only
  Qwen_300          real + 300 Qwen 14B single-shot synth
  GemFlash_300      real + 300 Gemini 2.5 Flash synth
  GemFL_300         real + 300 Gemini 2.5 Flash-Lite synth
  Gem31FL_300       real + 300 Gemini 3.1 Flash-Lite synth
  Claude_50         real + 50 Claude-authored posts
  OASIS_Qwen_300    real + 300 OASIS+Qwen posts (subsample of 672)
  OASIS_Gem_300     real + 300 OASIS+Gemini posts (subsample of 572)
  EDA_4x            real + 4*n_real EDA-augmented variants
  Mixed_best        real + 100 Qwen + 100 Gem31FL + 50 Claude = 250 hand-mixed

Outputs results/llm_tier_grid_seeds.csv and llm_tier_grid_summary.csv.
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
SYNTH = A3 / "synth"

sys.path.insert(0, str(A3 / "code" / "synth"))
from eda_augment import eda_one

MODEL_NAME = "cardiffnlp/twitter-roberta-base-2022-154m"
SEEDS_DEFAULT = (42, 1337, 2024, 7, 88)
N_REALS_DEFAULT = (100, 250)


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


def sample(records, n, seed):
    if not records:
        return []
    if n >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), n, replace=False)
    return [records[i] for i in idx]


def eda_augment_real(real_df, augs_per_post, seed):
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    real_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    synth = {
        "qwen": load_jsonl(SYNTH / "synth_posts.jsonl"),
        "qwen35": load_jsonl(SYNTH / "synth_qwen35.jsonl"),
        "deepseek": load_jsonl(SYNTH / "synth_deepseek.jsonl"),
        "gem_flash": load_jsonl(SYNTH / "smoke_g25flash.jsonl"),
        "gem_fl": load_jsonl(SYNTH / "smoke_g25flashlite.jsonl"),
        "gem_31fl": load_jsonl(SYNTH / "smoke_g31flashlite.jsonl"),
        "gem_fewshot": load_jsonl(SYNTH / "synth_gemini_fewshot.jsonl"),
        "claude": load_jsonl(SYNTH / "claude_authored.jsonl"),
        "oasis_qwen": load_jsonl(SYNTH / "oasis_posts_only.jsonl"),
        "oasis_gem": load_jsonl(SYNTH / "oasis_gemini_60.jsonl"),
    }
    for k, v in synth.items():
        print(f"  synth[{k}] = {len(v)} posts")

    rows = []
    out_path = RESULTS / "llm_tier_grid_seeds.csv"

    for n_real in args.n_reals:
        for seed in args.seeds:
            real_sub = real_df.groupby("label", group_keys=False).sample(
                n=n_real // 2, random_state=seed
            ).reset_index(drop=True)
            real_texts = real_sub["text"].tolist()
            real_labels = real_sub["label"].tolist()

            cells = []
            cells.append(("TRTR", real_texts, real_labels))

            def with_extras(name, extras):
                cells.append((name,
                              real_texts + [r["text"] for r in extras],
                              real_labels + [r["label"] for r in extras]))

            with_extras("Qwen25_300", sample(synth["qwen"], 300, seed))
            with_extras("Qwen35_300", sample(synth["qwen35"], 300, seed))
            with_extras("DeepSeek_300", sample(synth["deepseek"], 300, seed))
            with_extras("GemFlash_300", sample(synth["gem_flash"], 300, seed))
            with_extras("GemFL_300", sample(synth["gem_fl"], 300, seed))
            with_extras("Gem31FL_300", sample(synth["gem_31fl"], 300, seed))
            with_extras("GemFL_FewShot_300", sample(synth["gem_fewshot"], 300, seed))
            with_extras("Claude_50", sample(synth["claude"], 50, seed))
            with_extras("OASIS_Qwen_300", sample(synth["oasis_qwen"], 300, seed))
            with_extras("OASIS_Gem_300", sample(synth["oasis_gem"], 300, seed))

            # EDA: 4*n_real perturbations of the same real subset
            aug_t, aug_l = eda_augment_real(real_sub, 4, seed)
            cells.append(("EDA_4x", real_texts + aug_t, real_labels + aug_l))

            # Mixed: 100 Qwen + 100 Gem 3.1 FL + 50 Claude = 250 synth posts
            mix = (sample(synth["qwen"], 100, seed)
                   + sample(synth["gem_31fl"], 100, seed)
                   + sample(synth["claude"], 50, seed))
            with_extras("Mixed_best", mix)

            for cell_name, tr_texts, tr_labels in cells:
                t0 = time.time()
                macro_f1, acc, per_class = train_eval(
                    tr_texts, tr_labels, test_texts, test_labels, seed, device=device,
                )
                elapsed = time.time() - t0
                row = {
                    "n_real": n_real,
                    "cell": cell_name,
                    "seed": seed,
                    "n_train": len(tr_texts),
                    "macro_f1": macro_f1,
                    "accuracy": acc,
                    "f1_nonstress": per_class[0],
                    "f1_stress": per_class[1],
                    "seconds": elapsed,
                }
                rows.append(row)
                print(f"n_real={n_real} cell={cell_name:14s} seed={seed} "
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
    summary.to_csv(RESULTS / "llm_tier_grid_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
