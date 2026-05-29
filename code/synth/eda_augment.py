"""EDA augmentation (Wei & Zou 2019).

Four cheap rule-based augmentations: random synonym replacement, random
insertion, random swap, random deletion. The original paper finds these
yield consistent F1 gains on text classification, particularly in the
low-resource regime (training on 50% of data + EDA matches training on
100% of data without).

This implementation uses NLTK WordNet for synonyms; if NLTK is not present
or WordNet has no synonym, the operation falls back to random swap.

Usage:
    python code/synth/eda_augment.py --in CORPORA/.../dreaddit-train.csv \
        --out synth/eda_aug_n100_x4.jsonl --n_real 100 --augs_per_post 4
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import random
import re

import pandas as pd


def _ensure_wordnet():
    try:
        import nltk
        nltk.data.find("corpora/wordnet")
        return True
    except Exception:
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            return True
        except Exception:
            return False


_HAVE_WORDNET = _ensure_wordnet()


def get_synonyms(word):
    if not _HAVE_WORDNET:
        return []
    from nltk.corpus import wordnet
    syns = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            s = lem.name().replace("_", " ").lower()
            if s != word.lower() and s.isalpha():
                syns.add(s)
    return list(syns)


STOPWORDS = set("a an the and or but if while because as is am are was were be been being "
                "have has had do does did i you he she it we they me him her us them my "
                "your his its our their this that these those to of for with at by from "
                "on in into out up down over under not no yes do don't didn't can't won't "
                "i'm i've i'll i'd you're you've you'll he's she's it's we're we've they're "
                "what when where why how which who whom whose so".split())


def _tokenize(text):
    return re.findall(r"[A-Za-z']+|[^A-Za-z']", text)


def _detokenize(tokens):
    out = []
    for i, t in enumerate(tokens):
        if i > 0 and re.match(r"[A-Za-z']+", t) and re.match(r"[A-Za-z']+", tokens[i - 1]):
            out.append(" ")
        out.append(t)
    return "".join(out)


def synonym_replacement(words, n, rng):
    words = list(words)
    candidates = [w for w in words if w.lower() not in STOPWORDS and re.match(r"[A-Za-z']+", w)]
    rng.shuffle(candidates)
    num_replaced = 0
    for word in candidates:
        syns = get_synonyms(word)
        if syns:
            synonym = rng.choice(syns)
            for i, w in enumerate(words):
                if w == word:
                    words[i] = synonym
                    break
            num_replaced += 1
        if num_replaced >= n:
            break
    return words


def random_swap(words, n, rng):
    words = list(words)
    n = min(n, max(1, len(words) // 2))
    for _ in range(n):
        idx = [i for i, w in enumerate(words) if re.match(r"[A-Za-z']+", w)]
        if len(idx) < 2:
            break
        i, j = rng.sample(idx, 2)
        words[i], words[j] = words[j], words[i]
    return words


def random_deletion(words, p, rng):
    if not words:
        return words
    out = []
    for w in words:
        if re.match(r"[A-Za-z']+", w) and rng.random() < p:
            continue
        out.append(w)
    if not out:
        return [rng.choice(words)]
    return out


def random_insertion(words, n, rng):
    words = list(words)
    candidates = [w for w in words if w.lower() not in STOPWORDS and re.match(r"[A-Za-z']+", w)]
    for _ in range(n):
        if not candidates:
            break
        word = rng.choice(candidates)
        syns = get_synonyms(word)
        if not syns:
            continue
        synonym = rng.choice(syns)
        pos = rng.randint(0, len(words))
        words.insert(pos, synonym)
    return words


def eda_one(text, alpha=0.1, rng=None):
    """Apply one randomly-chosen EDA op to `text`. alpha controls op intensity."""
    rng = rng or random.Random()
    tokens = _tokenize(text)
    n = max(1, int(alpha * len([t for t in tokens if re.match(r"[A-Za-z']+", t)])))
    op = rng.choice(["sr", "ri", "rs", "rd"])
    if op == "sr":
        out = synonym_replacement(tokens, n, rng)
    elif op == "ri":
        out = random_insertion(tokens, n, rng)
    elif op == "rs":
        out = random_swap(tokens, n, rng)
    else:
        out = random_deletion(tokens, alpha, rng)
    return _detokenize(out), op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default=str(Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\CORPORA\REDDIT\DREADDIT\dreaddit-train.csv")))
    parser.add_argument("--n_real", type=int, default=500)
    parser.add_argument("--augs_per_post", type=int, default=4,
                        help="How many augmented variants to produce per real post.")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_csv(args.source_csv)
    if args.n_real and args.n_real < len(df):
        df = df.groupby("label", group_keys=False).sample(
            n=args.n_real // 2, random_state=args.seed
        ).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    op_counts = {"sr": 0, "ri": 0, "rs": 0, "rd": 0}
    with out_path.open("w", encoding="utf-8") as fh:
        for _, r in df.iterrows():
            for k in range(args.augs_per_post):
                aug_text, op = eda_one(r["text"], alpha=args.alpha, rng=rng)
                op_counts[op] += 1
                rec = {
                    "synth_post_id": f"eda_{r['post_id']}_{k}",
                    "text": aug_text,
                    "persona": "eda_real_paraphrase",
                    "label": int(r["label"]),
                    "simulation_id": "eda",
                    "round": k,
                    "variant": k,
                    "topic": "eda_augmentation",
                    "n_chars": len(aug_text),
                    "provider": "eda",
                    "model": f"eda_alpha{args.alpha}",
                    "source_post_id": r["post_id"],
                    "eda_op": op,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    print(f"wrote {kept} EDA-augmented posts to {out_path}")
    print(f"op histogram: {op_counts}")
    print(f"WordNet available: {_HAVE_WORDNET}")


if __name__ == "__main__":
    main()
