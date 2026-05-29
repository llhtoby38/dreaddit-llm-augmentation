"""Gemini-backed variant of generate_synth.py.

Posts the same persona/topic grid to the Google Generative-Language REST API instead of
Ollama. The output JSONL schema is byte-compatible with generate_synth.py, so downstream
audit and grid scripts work unchanged.

Run with:
    setx GEMINI_API_KEY "..."  (once)
    python code/synth/generate_synth_gemini.py --model gemini-2.5-flash --target 300 --out synth/smoke_gemini-2.5-flash.jsonl

The script reads GEMINI_API_KEY from environment (or from an Assignment-3/.env.local file).
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

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from personas import PERSONAS

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
SYNTH_DIR = A3 / "synth"
SYNTH_DIR.mkdir(exist_ok=True)
LOG_DIR = A3 / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


ROUGHENING_PREAMBLE = (
    "Style note (very important): write the way an actual Reddit user would on "
    "their phone at 11pm. Casual register. Lowercase is fine. Short sentences "
    "mixed with run-on sentences. Occasional abbreviations like 'ur', 'tbh', "
    "'rn', 'idk', 'lol' where natural. Do not over-do typos. Avoid polished, "
    "blog-style prose; avoid clever neologisms; avoid lists with bullet points. "
    "Sometimes start with a lowercase 'so' or 'idk' or 'ok'. Vary sentence "
    "length and let some sentences trail off. You are venting, not writing for "
    "publication.\n\n"
)


def load_api_key():
    """Load GEMINI_API_KEY from env, falling back to Assignment-3/.env.local."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    env_path = A3 / ".env.local"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("Set GEMINI_API_KEY environment variable or write to .env.local")


def gemini_generate(api_key, model, system, prompt, temperature, top_p, max_tokens,
                    timeout=180, retries=4, backoff=2.0):
    # thinkingBudget choice:
    #   - Pro models require thinking mode; we set -1 (auto) to let the API choose.
    #   - Flash / Flash-Lite default to thinking but accept 0 to disable.
    # The maxOutputTokens cap is separate from the thinking budget.
    is_pro = "pro" in model.lower()
    thinking_budget = -1 if is_pro else 0
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": thinking_budget},
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
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
            body_text = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            if e.code in (429, 500, 502, 503, 504):
                sleep = backoff * (2 ** attempt) + random.random()
                time.sleep(min(60, sleep))
                continue
            raise RuntimeError(f"HTTP {e.code}: {body_text[:200]}") from e
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** attempt))
            continue
    raise RuntimeError(f"Gemini call failed after {retries} retries: {last_err}")


# --- post-processing (mirror Ollama generator) -------------------------------

WRAP_PATTERNS = [
    re.compile(r"^\s*(here(?:'s| is) (?:a|the|my)?[^\n:]*:)\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*+title\*+\s*:?\s*[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*title\s*:\s*[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*+[^*\n]+\*+\s*\n", re.MULTILINE),
    re.compile(r"^\s*\[[A-Za-z ]{2,40}\]\s*\n", re.MULTILINE),
]
LEADING_QUOTE = re.compile(r'^\s*["“”‘’]+|["“”‘’]+\s*$')

SLIP_PATTERNS = [
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bI(?:'m| am) (?:a |an )?(?:language model|AI assistant|AI)\b", re.IGNORECASE),
    re.compile(r"\bI cannot (?:provide|generate)\b", re.IGNORECASE),
    re.compile(r"\bsure[,!]? here(?:'s| is)\b", re.IGNORECASE),
    re.compile(r"\bcertainly[,!]? here(?:'s| is)\b", re.IGNORECASE),
]


def clean_post(text):
    text = text.strip()
    for pat in WRAP_PATTERNS:
        text = pat.sub("", text, count=1).strip()
    text = LEADING_QUOTE.sub("", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def is_persona_slip(text):
    return any(p.search(text) for p in SLIP_PATTERNS)


def jaccard5(a, b):
    def grams(s):
        s = re.sub(r"\s+", " ", s.lower())
        return {s[i:i + 5] for i in range(len(s) - 4)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def build_user_prompt(persona, topic, rnd):
    variants = [
        f"Write a single Reddit post (~3-7 sentences) about: {topic}. Do NOT include a title, hashtags, or sign-off. Just the post body. Stay strictly in your character; never break the fourth wall.",
        f"Compose a short Reddit post by yourself about: {topic}. Keep it to one paragraph or two short paragraphs. No title, no metadata, no 'sure here is'. Just the body of the post.",
        f"You are about to write a Reddit post about: {topic}. Write the post body only. Three to seven sentences. Do not preface, do not title, do not sign off. Stay in character.",
        f"Reddit post body, your own voice, topic: {topic}. No title, no preamble. 3-7 sentences. Stay strictly in your persona.",
    ]
    return variants[rnd % len(variants)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Gemini model name, e.g. gemini-2.5-flash")
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--simulations", type=int, default=3)
    parser.add_argument("--variants_per_topic", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=260)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_chars", type=int, default=2400)
    parser.add_argument("--dedup_threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start_simulation", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--inter_call_sleep", type=float, default=0.0,
                        help="Optional sleep between calls to stay under free-tier RPM.")
    parser.add_argument("--roughen", action="store_true",
                        help="Prepend a Reddit-register preamble to each persona's system prompt.")
    args = parser.parse_args()

    api_key = load_api_key()
    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"synth_gen_gemini_{args.model.replace('/', '_')}.log"

    grid = []
    sim_lo = args.start_simulation
    sim_hi = args.start_simulation + args.simulations
    for sim in range(sim_lo, sim_hi):
        for persona in PERSONAS:
            topics = list(persona["topics"])
            random.Random(args.seed + sim).shuffle(topics)
            for ti, topic in enumerate(topics):
                for v in range(args.variants_per_topic):
                    grid.append({
                        "simulation_id": sim,
                        "persona": persona["name"],
                        "label": persona["label"],
                        "topic": topic,
                        "round": ti * args.variants_per_topic + v,
                        "variant": v,
                    })
    rng.shuffle(grid)
    print(f"grid={len(grid)} model={args.model} target={args.target} out={out_path}")

    kept_texts = []
    kept_records = []
    dropped = {"short": 0, "long": 0, "slip": 0, "dup": 0, "error": 0}
    n_calls = 0
    total_eval_tokens = 0
    total_latency = 0.0
    t0_all = time.time()
    persona_lookup = {p["name"]: p for p in PERSONAS}

    fh_mode = "a" if args.append else "w"
    log_mode = "a" if args.append else "w"
    if args.append and out_path.exists():
        with out_path.open(encoding="utf-8") as in_fh:
            for line in in_fh:
                try:
                    rec = json.loads(line)
                    kept_texts.append(rec["text"])
                    kept_records.append(rec)
                except Exception:
                    pass
        print(f"appending: existing kept = {len(kept_records)}")

    with out_path.open(fh_mode, encoding="utf-8") as fh, log_path.open(log_mode, encoding="utf-8") as logfh:
        logfh.write(f"start={time.strftime('%Y-%m-%d %H:%M:%S')} model={args.model} target={args.target}\n")
        logfh.flush()
        for cell in grid:
            if len(kept_records) >= args.target:
                break
            persona = persona_lookup[cell["persona"]]
            if args.roughen:
                persona = dict(persona)
                persona["system"] = ROUGHENING_PREAMBLE + persona["system"]
            user_prompt = build_user_prompt(persona, cell["topic"], cell["round"])
            t0 = time.time()
            try:
                raw_text, eval_count = gemini_generate(
                    api_key, args.model, persona["system"], user_prompt,
                    args.temperature, args.top_p, args.max_tokens,
                )
            except Exception as exc:
                dropped["error"] += 1
                logfh.write(f"ERROR {cell['persona']} topic={cell['topic']!r} -> {exc}\n")
                logfh.flush()
                if args.inter_call_sleep:
                    time.sleep(args.inter_call_sleep)
                continue
            dt = time.time() - t0
            n_calls += 1
            total_eval_tokens += eval_count
            total_latency += dt

            text = clean_post(raw_text)
            if is_persona_slip(text):
                dropped["slip"] += 1
                if args.inter_call_sleep:
                    time.sleep(args.inter_call_sleep)
                continue
            if len(text) < args.min_chars:
                dropped["short"] += 1
                if args.inter_call_sleep:
                    time.sleep(args.inter_call_sleep)
                continue
            if len(text) > args.max_chars:
                dropped["long"] += 1
                if args.inter_call_sleep:
                    time.sleep(args.inter_call_sleep)
                continue
            is_dup = False
            for prev in kept_texts[-200:]:
                if jaccard5(text, prev) >= args.dedup_threshold:
                    is_dup = True
                    break
            if is_dup:
                dropped["dup"] += 1
                if args.inter_call_sleep:
                    time.sleep(args.inter_call_sleep)
                continue

            post_id = hashlib.sha1(
                f"{args.model}|{cell['simulation_id']}|{cell['persona']}|{cell['topic']}|{cell['variant']}|{text[:40]}".encode("utf-8")
            ).hexdigest()[:16]
            record = {
                "synth_post_id": post_id,
                "text": text,
                "persona": cell["persona"],
                "label": cell["label"],
                "simulation_id": cell["simulation_id"],
                "round": cell["round"],
                "variant": cell["variant"],
                "topic": cell["topic"],
                "n_chars": len(text),
                "provider": "gemini",
                "model": args.model,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            kept_records.append(record)
            kept_texts.append(text)

            if n_calls % 25 == 0:
                tok_per_s = total_eval_tokens / total_latency if total_latency else 0
                msg = (f"model={args.model} calls={n_calls} kept={len(kept_records)} "
                       f"dropped={dropped} tok/s={tok_per_s:.1f} "
                       f"elapsed={time.time()-t0_all:.0f}s")
                print(msg)
                logfh.write(msg + "\n")
                logfh.flush()

            if args.inter_call_sleep:
                time.sleep(args.inter_call_sleep)

    elapsed = time.time() - t0_all
    summary = {
        "model": args.model,
        "kept": len(kept_records),
        "n_calls": n_calls,
        "dropped": dropped,
        "tok_per_s_overall": total_eval_tokens / total_latency if total_latency else 0,
        "elapsed_sec": elapsed,
        "out_path": str(out_path),
    }
    summary_path = LOG_DIR / f"synth_gen_gemini_{args.model.replace('/', '_')}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
