"""Stage 1 baseline: fine-tune MentalBERT on real DREDDIT, evaluate on real test split.

Run with:
    python code/baseline/run_baseline.py

Outputs:
    results/baseline_seeds.csv          one row per seed, macro-F1 + accuracy
    results/baseline_summary.csv        mean + std + bootstrap 95% CI
    outputs/preds/baseline_seed{S}.csv  test predictions per seed, for subgroup analysis
    outputs/logs/baseline_seed{S}.log   per-epoch training loss

Mirrors Assignment 2B Part 6 idioms: AutoTokenizer + AutoModelForSequenceClassification,
manual training loop, AdamW, no Trainer API.
"""

from pathlib import Path
import argparse
import csv
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DREDDIT_TRAIN = ROOT / "CORPORA"
# Repo layout uses CORPORA outside Assignment 3/. Fall back to the project-root version.
if not (DREDDIT_TRAIN / "REDDIT").exists():
    DREDDIT_TRAIN = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\CORPORA")
TRAIN_CSV = DREDDIT_TRAIN / "REDDIT" / "DREADDIT" / "dreaddit-train.csv"
TEST_CSV = DREDDIT_TRAIN / "REDDIT" / "DREADDIT" / "dreaddit-test.csv"

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
RESULTS = A3 / "results"
PRED_DIR = A3 / "outputs" / "preds"
LOG_DIR = A3 / "outputs" / "logs"
for p in (RESULTS, PRED_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-2022-154m"
FALLBACK_MODEL = "distilbert-base-uncased"
SEEDS = (42, 1337, 2024, 7, 88)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_texts(texts, tokenizer, max_length):
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def train_one_seed(seed, train_texts, train_labels, test_texts, test_labels,
                   model_name, max_length, batch_size, lr, epochs, device, log_path):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_ids, train_mask = encode_texts(train_texts, tokenizer, max_length)
    test_ids, test_mask = encode_texts(test_texts, tokenizer, max_length)
    train_ds = TensorDataset(train_ids, train_mask, torch.tensor(train_labels))
    test_ds = TensorDataset(test_ids, test_mask, torch.tensor(test_labels))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optim = AdamW(model.parameters(), lr=lr)
    log_lines = [f"seed={seed} model={model_name} max_len={max_length} bs={batch_size} lr={lr} epochs={epochs}"]
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_n = 0
        for batch in tqdm(train_loader, desc=f"seed{seed} ep{epoch+1}/{epochs}", leave=False):
            ids, mask, labels = (t.to(device) for t in batch)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            running_loss += out.loss.item() * labels.size(0)
            running_n += labels.size(0)
        avg_loss = running_loss / running_n
        log_lines.append(f"epoch {epoch+1}/{epochs} train_loss={avg_loss:.4f}")

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids, mask, _ = (t.to(device) for t in batch)
            logits = model(input_ids=ids, attention_mask=mask).logits
            soft = torch.softmax(logits, dim=-1)
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            probs.extend(soft[:, 1].cpu().numpy().tolist())

    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    return np.array(preds), np.array(probs)


def bootstrap_ci(y_true, y_pred, n=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro")
    lo = float(np.quantile(scores, alpha / 2))
    hi = float(np.quantile(scores, 1 - alpha / 2))
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME, help="HF checkpoint name")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--tag", default="baseline", help="Output filename prefix")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"train={len(train_df)} test={len(test_df)}")

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    seed_rows = []
    for seed in args.seeds:
        t0 = time.time()
        log_path = LOG_DIR / f"{args.tag}_seed{seed}.log"
        preds, probs = train_one_seed(
            seed, train_texts, train_labels, test_texts, test_labels,
            args.model, args.max_length, args.batch_size, args.lr, args.epochs, device, log_path,
        )
        macro_f1 = f1_score(test_labels, preds, average="macro")
        acc = accuracy_score(test_labels, preds)
        per_class = f1_score(test_labels, preds, average=None)
        elapsed = time.time() - t0
        print(f"seed={seed} macro_f1={macro_f1:.4f} acc={acc:.4f} f1_0={per_class[0]:.4f} f1_1={per_class[1]:.4f} time={elapsed:.0f}s")

        pred_path = PRED_DIR / f"{args.tag}_seed{seed}.csv"
        pred_df = pd.DataFrame({
            "post_id": test_df["post_id"],
            "subreddit": test_df["subreddit"],
            "text_len": test_df["text"].str.len(),
            "label": test_labels,
            "pred": preds,
            "prob_stress": probs,
        })
        pred_df.to_csv(pred_path, index=False)
        seed_rows.append({
            "tag": args.tag,
            "seed": seed,
            "macro_f1": macro_f1,
            "accuracy": acc,
            "f1_nonstress": per_class[0],
            "f1_stress": per_class[1],
            "seconds": elapsed,
        })

    df = pd.DataFrame(seed_rows)
    seeds_csv = RESULTS / f"{args.tag}_seeds.csv"
    df.to_csv(seeds_csv, index=False)
    print(f"wrote {seeds_csv}")

    # mean + std over seeds; bootstrap CI uses seed 42's predictions only (representative)
    rep_seed = args.seeds[0]
    rep_preds = pd.read_csv(PRED_DIR / f"{args.tag}_seed{rep_seed}.csv")
    lo, hi = bootstrap_ci(rep_preds["label"].to_numpy(), rep_preds["pred"].to_numpy())
    summary = {
        "tag": args.tag,
        "n_seeds": len(args.seeds),
        "macro_f1_mean": float(df["macro_f1"].mean()),
        "macro_f1_std": float(df["macro_f1"].std()),
        "accuracy_mean": float(df["accuracy"].mean()),
        "boot_ci_lo": lo,
        "boot_ci_hi": hi,
        "boot_ci_basis_seed": rep_seed,
    }
    summary_csv = RESULTS / f"{args.tag}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
