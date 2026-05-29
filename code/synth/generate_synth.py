"""Stage 2 synthetic-corpus generator (persona-pool, Stage 0b in the handoff).

For each persona x simulation x round x variant, prompts Qwen2.5-14B-Instruct via Ollama
to write a single Reddit-style post. The label comes from the persona's archetype.

This script implements the Stage 0b fallback described in the handoff. It strips MiroFish's
GraphRAG and Zep memory layer but preserves the multi-persona LLM-data-generator core that
the research question tests.

Run with:
    python code/synth/generate_synth.py --target 5000 --model qwen2.5:14b-instruct-q4_K_M
"""

from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from personas import PERSONAS

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
SYNTH_DIR = A3 / "synth"
SYNTH_DIR.mkdir(exist_ok=True)
LOG_DIR = A3 / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "qwen2.5:14b-instruct-q4_K_M"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Register-roughening preamble — prepended to each persona's system prompt when
# --roughen is set. Calibrated against a manual inspection of DREDDIT samples:
# real posts skew lowercase, abbreviated, typo-tolerant, and stop short of
# polished paragraphs. We push the LLM in that direction without specifying
# typos verbatim (which the LLM tends to over-execute).
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


def ollama_generate(model, system, prompt, temperature, top_p, max_tokens, timeout=180):
    # think=false disables R1-style reasoning blocks; harmless for non-reasoning models.
    body = json.dumps({
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# --- post-processing ---------------------------------------------------------

WRAP_PATTERNS = [
    re.compile(r"^\s*(here(?:'s| is) (?:a|the|my)?[^\n:]*:)\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*+title\*+\s*:?\s*[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*title\s*:\s*[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*+[^*\n]+\*+\s*\n", re.MULTILINE),  # bold-only headers
    re.compile(r"^\s*\[[A-Za-z ]{2,40}\]\s*\n", re.MULTILINE),  # [Post Title] lines
]
LEADING_QUOTE = re.compile(r'^\s*["“”‘’]+|["“”‘’]+\s*$')


def clean_post(text):
    """Strip common LLM wrappers: 'Here's a post:', leading titles, surrounding quotes."""
    text = text.strip()
    for pat in WRAP_PATTERNS:
        text = pat.sub("", text, count=1).strip()
    text = LEADING_QUOTE.sub("", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


SLIP_PATTERNS = [
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bI(?:'m| am) (?:a |an )?(?:language model|AI assistant|AI)\b", re.IGNORECASE),
    re.compile(r"\bI cannot (?:provide|generate)\b", re.IGNORECASE),
    re.compile(r"\bsure[,!]? here(?:'s| is)\b", re.IGNORECASE),
    re.compile(r"\bcertainly[,!]? here(?:'s| is)\b", re.IGNORECASE),
]


def is_persona_slip(text):
    return any(p.search(text) for p in SLIP_PATTERNS)


def jaccard5(a, b):
    """Character-5gram Jaccard. Cheap near-duplicate check."""
    def grams(s):
        s = re.sub(r"\s+", " ", s.lower())
        return {s[i:i + 5] for i in range(len(s) - 4)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


# --- generation loop ---------------------------------------------------------


def build_user_prompt(persona, topic, rnd):
    """Compose the user-turn prompt for a single post."""
    variants = [
        f"Write a single Reddit post (~3-7 sentences) about: {topic}. Do NOT include a title, hashtags, or sign-off. Just the post body. Stay strictly in your character; never break the fourth wall.",
        f"Compose a short Reddit post by yourself about: {topic}. Keep it to one paragraph or two short paragraphs. No title, no metadata, no 'sure here is'. Just the body of the post.",
        f"You are about to write a Reddit post about: {topic}. Write the post body only. Three to seven sentences. Do not preface, do not title, do not sign off. Stay in character.",
        f"Reddit post body, your own voice, topic: {topic}. No title, no preamble. 3-7 sentences. Stay strictly in your persona.",
    ]
    return variants[rnd % len(variants)]


def gen_one(model, persona, topic, rnd, temperature, top_p, max_tokens):
    """Generate one cleaned post. Returns (text, was_slipped, n_tokens, latency_s)."""
    user_prompt = build_user_prompt(persona, topic, rnd)
    t0 = time.time()
    j = ollama_generate(model, persona["system"], user_prompt, temperature, top_p, max_tokens)
    dt = time.time() - t0
    raw = j.get("response", "")
    text = clean_post(raw)
    slipped = is_persona_slip(text)
    eval_count = j.get("eval_count", 0)
    return text, slipped, eval_count, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target", type=int, default=5000, help="Target post count after gating.")
    parser.add_argument("--simulations", type=int, default=3)
    parser.add_argument("--variants_per_topic", type=int, default=4,
                        help="Resampled variations per (persona, topic) within a simulation.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=260)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_chars", type=int, default=2400)
    parser.add_argument("--dedup_threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--out", default=str(SYNTH_DIR / "synth_posts.jsonl"))
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N successful posts. 0 = run full grid.")
    parser.add_argument("--append", action="store_true",
                        help="Append to --out instead of overwriting. Useful for two-pass runs.")
    parser.add_argument("--start_simulation", type=int, default=0,
                        help="First simulation_id to use. With --append set this past the largest "
                             "simulation_id already in --out to avoid reused topic-orders.")
    parser.add_argument("--roughen", action="store_true",
                        help="Prepend a Reddit-register preamble to each persona's system prompt.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    log_path = LOG_DIR / "synth_gen.log"

    # Pre-compose the generation grid.
    grid = []
    sim_lo = args.start_simulation
    sim_hi = args.start_simulation + args.simulations
    for sim in range(sim_lo, sim_hi):
        for persona in PERSONAS:
            topics = list(persona["topics"])
            # Use a sim-derived seed so shuffles differ between simulations even with
            # a fixed master seed.
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
    print(f"grid_size={len(grid)} sims=[{sim_lo},{sim_hi}) target={args.target} model={args.model}")

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
    # When appending, seed kept_texts from the existing file so the dedup window sees prior posts.
    if args.append and out_path.exists():
        with out_path.open(encoding="utf-8") as in_fh:
            for line in in_fh:
                try:
                    rec = json.loads(line)
                    kept_texts.append(rec["text"])
                    kept_records.append(rec)
                except Exception:
                    continue
        print(f"appending: existing kept = {len(kept_records)}")
    with out_path.open(fh_mode, encoding="utf-8") as fh, log_path.open(log_mode, encoding="utf-8") as logfh:
        logfh.write(f"start={time.strftime('%Y-%m-%d %H:%M:%S')} target={args.target}\n")
        logfh.flush()
        for cell in grid:
            if args.limit and len(kept_records) >= args.limit:
                break
            if len(kept_records) >= args.target:
                break
            persona = persona_lookup[cell["persona"]]
            if args.roughen:
                # Don't mutate the shared object; create a per-call view.
                persona = dict(persona)
                persona["system"] = ROUGHENING_PREAMBLE + persona["system"]
            try:
                text, slipped, eval_count, dt = gen_one(
                    args.model, persona, cell["topic"], cell["round"],
                    args.temperature, args.top_p, args.max_tokens,
                )
            except Exception as exc:
                dropped["error"] += 1
                logfh.write(f"ERROR {cell['persona']} topic={cell['topic']!r} -> {exc}\n")
                logfh.flush()
                continue
            n_calls += 1
            total_eval_tokens += eval_count
            total_latency += dt

            if slipped:
                dropped["slip"] += 1
                continue
            if len(text) < args.min_chars:
                dropped["short"] += 1
                continue
            if len(text) > args.max_chars:
                dropped["long"] += 1
                continue
            is_dup = False
            for prev in kept_texts[-200:]:
                if jaccard5(text, prev) >= args.dedup_threshold:
                    is_dup = True
                    break
            if is_dup:
                dropped["dup"] += 1
                continue

            post_id = hashlib.sha1(f"{cell['simulation_id']}|{cell['persona']}|{cell['topic']}|{cell['variant']}|{text[:40]}".encode("utf-8")).hexdigest()[:16]
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
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            kept_records.append(record)
            kept_texts.append(text)

            if n_calls % 25 == 0:
                tok_per_s = total_eval_tokens / total_latency if total_latency else 0
                elapsed = time.time() - t0_all
                msg = (
                    f"calls={n_calls} kept={len(kept_records)} dropped={dropped} "
                    f"tok/s={tok_per_s:.1f} elapsed={elapsed:.0f}s"
                )
                print(msg)
                logfh.write(msg + "\n")
                logfh.flush()

    elapsed = time.time() - t0_all
    summary = {
        "kept": len(kept_records),
        "n_calls": n_calls,
        "dropped": dropped,
        "tok_per_s_overall": total_eval_tokens / total_latency if total_latency else 0,
        "elapsed_sec": elapsed,
        "out_path": str(out_path),
    }
    (LOG_DIR / "synth_gen_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
