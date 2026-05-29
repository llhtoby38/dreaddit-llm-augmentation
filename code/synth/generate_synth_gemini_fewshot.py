"""Gemini generator with REAL Dreaddit examples as few-shot in-context anchors.

For each persona, sample three label-matched real Dreaddit posts and include them
in the user-turn prompt as concrete examples of the target register. This tests
Sahu et al. [21]'s positive finding: LLM augmentation helps when the target
distribution is shown explicitly.

Output schema is JSONL, byte-compatible with the other generators so the same
audit + grid scripts handle it unchanged.

Run with:
    python code/synth/generate_synth_gemini_fewshot.py --model gemini-2.5-flash-lite --target 500
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error

import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from personas import PERSONAS

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
ROOT = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090")
TRAIN_CSV = ROOT / "CORPORA" / "REDDIT" / "DREADDIT" / "dreaddit-train.csv"
SYNTH_DIR = A3 / "synth"
LOG_DIR = A3 / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    env_path = A3 / ".env.local"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("Set GEMINI_API_KEY")


def gemini_generate(api_key, model, system, prompt, temperature, top_p, max_tokens,
                    timeout=120, retries=4, backoff=2.0):
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                j = json.loads(resp.read())
            cands = j.get("candidates") or []
            if not cands:
                return "", 0
            parts = cands[0].get("content", {}).get("parts", []) or []
            text = "".join(p.get("text", "") for p in parts)
            usage = j.get("usageMetadata", {})
            eval_count = usage.get("candidatesTokenCount", 0) or 0
            return text, eval_count
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (2 ** attempt) + random.random())
                continue
            body_text = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"HTTP {e.code}: {body_text[:200]}") from e
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Gemini call failed after {retries} retries: {last_err}")


WRAP_PATTERNS = [
    re.compile(r"^\s*(here(?:'s| is) (?:a|the|my)?[^\n:]*:)\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*title\s*:\s*[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*+[^*\n]+\*+\s*\n", re.MULTILINE),
]
SLIP = [
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bI(?:'m| am) (?:a |an )?(?:language model|AI)\b", re.IGNORECASE),
    re.compile(r"\bsure[,!]? here(?:'s| is)\b", re.IGNORECASE),
    re.compile(r"\bexample\s*\d+\s*:?", re.IGNORECASE),  # avoid echoing back "Example 1:" etc.
]


def clean_post(text):
    text = (text or "").strip()
    for p in WRAP_PATTERNS:
        text = p.sub("", text, count=1).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def is_slip(text):
    return any(p.search(text) for p in SLIP)


def jaccard5(a, b):
    def grams(s):
        s = re.sub(r"\s+", " ", s.lower())
        return {s[i:i + 5] for i in range(len(s) - 4)}
    ga, gb = grams(a), grams(b)
    return (len(ga & gb) / len(ga | gb)) if ga and gb else 0.0


def sample_real_examples(real_df, label, k, persona_subreddit_hint, rng):
    """Pick k real Dreaddit posts matching label; prefer subreddit-aligned posts if hint matches."""
    pool = real_df[real_df["label"] == label]
    hint_subreddits = [s.strip() for s in re.split(r"[,/]", persona_subreddit_hint or "") if s.strip()]
    aligned = pool[pool["subreddit"].isin(hint_subreddits)] if hint_subreddits else pool
    if len(aligned) >= k:
        idx = rng.sample(range(len(aligned)), k)
        return aligned.iloc[idx]["text"].tolist()
    # Fall back to any matching-label post
    idx = rng.sample(range(len(pool)), min(k, len(pool)))
    return pool.iloc[idx]["text"].tolist()


def build_fewshot_prompt(persona, topic, real_examples):
    examples_block = "\n\n".join(
        f"Example {i+1} (real Reddit post in this register):\n{ex.strip()}"
        for i, ex in enumerate(real_examples)
    )
    return (
        "Below are three real Reddit posts in the register you should mimic. "
        "Notice the fragmented sentence structure, casual lowercase usage, "
        "abrupt topic shifts, occasional typos, mid-thought endings, and lack "
        "of polished blog-style organisation. Do NOT copy the examples; write "
        "your own post in this same register.\n\n"
        f"{examples_block}\n\n"
        f"Now you, in your assigned persona, write a single Reddit post about: "
        f"{topic}. 3-7 sentences. No title, no preamble, no hashtags, no "
        "@-mentions. Just the post body."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--variants_per_topic", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=260)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_chars", type=int, default=2400)
    parser.add_argument("--dedup_threshold", type=float, default=0.8)
    parser.add_argument("--examples_per_call", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--out", default=str(SYNTH_DIR / "synth_gemini_fewshot.jsonl"))
    args = parser.parse_args()

    api_key = load_api_key()
    rng = random.Random(args.seed)
    out_path = Path(args.out)
    log_path = LOG_DIR / f"synth_gen_gemini_fewshot_{args.model.replace('/', '_')}.log"

    real_df = pd.read_csv(TRAIN_CSV)

    grid = []
    for sim in range(args.simulations):
        for persona in PERSONAS:
            topics = list(persona["topics"])
            random.Random(args.seed + sim).shuffle(topics)
            for ti, topic in enumerate(topics):
                for v in range(args.variants_per_topic):
                    grid.append({
                        "simulation_id": sim, "persona": persona["name"],
                        "label": persona["label"], "topic": topic,
                        "round": ti * args.variants_per_topic + v, "variant": v,
                    })
    rng.shuffle(grid)
    print(f"grid={len(grid)} model={args.model} target={args.target}")

    persona_lookup = {p["name"]: p for p in PERSONAS}
    kept_records, kept_texts = [], []
    dropped = {"short": 0, "long": 0, "slip": 0, "dup": 0, "error": 0}
    n_calls = 0
    total_eval = 0
    total_latency = 0.0
    t0_all = time.time()

    with out_path.open("w", encoding="utf-8") as fh, log_path.open("w", encoding="utf-8") as logfh:
        logfh.write(f"start={time.strftime('%Y-%m-%d %H:%M:%S')} model={args.model} target={args.target}\n")
        for cell in grid:
            if len(kept_records) >= args.target:
                break
            persona = persona_lookup[cell["persona"]]
            # Sample 3 real examples matching this persona's label
            real_examples = sample_real_examples(
                real_df, persona["label"], args.examples_per_call,
                persona.get("subreddit_hint", ""), rng,
            )
            user_prompt = build_fewshot_prompt(persona, cell["topic"], real_examples)
            t0 = time.time()
            try:
                raw, eval_count = gemini_generate(
                    api_key, args.model, persona["system"], user_prompt,
                    args.temperature, args.top_p, args.max_tokens,
                )
            except Exception as exc:
                dropped["error"] += 1
                logfh.write(f"ERROR {cell['persona']} -> {exc}\n")
                logfh.flush()
                continue
            dt = time.time() - t0
            n_calls += 1
            total_eval += eval_count
            total_latency += dt

            text = clean_post(raw)
            if is_slip(text):
                dropped["slip"] += 1
                continue
            if len(text) < args.min_chars:
                dropped["short"] += 1
                continue
            if len(text) > args.max_chars:
                dropped["long"] += 1
                continue
            is_dup = any(jaccard5(text, prev) >= args.dedup_threshold for prev in kept_texts[-200:])
            if is_dup:
                dropped["dup"] += 1
                continue

            post_id = hashlib.sha1(
                f"{args.model}|fewshot|{cell['simulation_id']}|{cell['persona']}|{cell['topic']}|{cell['variant']}|{text[:40]}".encode("utf-8")
            ).hexdigest()[:16]
            rec = {
                "synth_post_id": post_id, "text": text,
                "persona": cell["persona"], "label": cell["label"],
                "simulation_id": cell["simulation_id"], "round": cell["round"],
                "variant": cell["variant"], "topic": cell["topic"], "n_chars": len(text),
                "provider": "gemini_fewshot", "model": args.model,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            kept_records.append(rec)
            kept_texts.append(text)

            if n_calls % 25 == 0:
                msg = (f"model={args.model} calls={n_calls} kept={len(kept_records)} "
                       f"dropped={dropped} tok/s={total_eval/total_latency:.1f} "
                       f"elapsed={time.time()-t0_all:.0f}s")
                print(msg)
                logfh.write(msg + "\n")
                logfh.flush()

    print(f"DONE kept={len(kept_records)} elapsed={time.time()-t0_all:.0f}s")


if __name__ == "__main__":
    main()
