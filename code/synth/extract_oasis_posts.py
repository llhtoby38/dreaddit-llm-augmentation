"""Extract posts from an OASIS reddit_simulation.db and write them to our JSONL schema.

OASIS stores agent actions in a SQLite database. The schema is exposed by the
`oasis` package; this script reads from the `post` table (and joins to `user`)
to get the content, author, label (derived via profile mapping back into our
personas.py), and timestamp for each generated post.

Usage:
    python code/synth/extract_oasis_posts.py \
        --db external/MiroFish_runs/smoke/reddit_simulation.db \
        --profiles external/MiroFish_runs/smoke/reddit_profiles.json \
        --out synth/oasis_smoke.jsonl
"""

from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import re
import sqlite3
import sys

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from personas import PERSONAS

PERSONA_BY_NAME = {p["name"]: p for p in PERSONAS}


WRAP_PATTERNS = [
    re.compile(r"^\s*(here(?:'s| is) (?:a|the|my)?[^\n:]*:)\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*+title\*+\s*:?\s*[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*title\s*:\s*[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*+[^*\n]+\*+\s*\n", re.MULTILINE),
    re.compile(r"^\s*\[[A-Za-z ]{2,40}\]\s*\n", re.MULTILINE),
]


def clean_post(text):
    text = (text or "").strip()
    for pat in WRAP_PATTERNS:
        text = pat.sub("", text, count=1).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_chars", type=int, default=2400)
    parser.add_argument("--include_comments", action="store_true",
                        help="Also emit comments as separate JSONL records.")
    parser.add_argument("--skip_seeds", action="store_true",
                        help="Skip posts that match strings in initial_posts (re-loaded from "
                             "the simulation_config.json alongside --profiles).")
    args = parser.parse_args()

    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    # username -> persona archetype name (the user_name prefix is "{persona_name}_{user_id}")
    user_meta = {}
    for prof in profiles:
        uid = prof["user_id"]
        uname = prof["user_name"]
        # The archetype prefix is everything before the final _<id>.
        archetype = uname.rsplit("_", 1)[0]
        if archetype not in PERSONA_BY_NAME:
            archetype = prof["user_name"]
        label = PERSONA_BY_NAME.get(archetype, {}).get("label", -1)
        user_meta[uid] = {"persona": archetype, "label": label, "user_name": uname}

    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    # Discover schema.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"tables: {tables}")

    if "post" not in tables:
        print("No `post` table found; nothing to extract.")
        return

    cursor.execute("PRAGMA table_info(post)")
    post_cols = [r[1] for r in cursor.fetchall()]
    print(f"post columns: {post_cols}")

    # Build a set of seed-post content strings (so --skip_seeds can drop them).
    seed_contents = set()
    if args.skip_seeds:
        sim_cfg = Path(args.profiles).parent / "simulation_config.json"
        if sim_cfg.exists():
            try:
                cfg = json.loads(sim_cfg.read_text(encoding="utf-8"))
                for p in cfg.get("event_config", {}).get("initial_posts", []):
                    seed_contents.add(p.get("content", "").strip())
            except Exception as e:
                print(f"warning: could not load seed posts: {e}")

    sources = [("post", "post_id")]
    if args.include_comments and "comment" in tables:
        sources.append(("comment", "comment_id"))

    kept = 0
    kept_posts = 0
    kept_comments = 0
    skipped = {"short": 0, "long": 0, "missing_label": 0, "empty": 0, "seed": 0}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for tbl, pk in sources:
            cursor.execute(f"PRAGMA table_info({tbl})")
            cols = [r[1] for r in cursor.fetchall()]
            wanted = [c for c in (pk, "user_id", "content", "created_at",
                                  "post_id", "original_post_id") if c in cols]
            select_cols = ", ".join(wanted)
            cursor.execute(f"SELECT {select_cols} FROM {tbl}")
            rows = cursor.fetchall()
            print(f"raw {tbl}: {len(rows)}")
            idx = {c: i for i, c in enumerate(wanted)}
            for row in rows:
                content = row[idx["content"]] if "content" in idx else ""
                user_id = row[idx["user_id"]] if "user_id" in idx else None
                this_id = row[idx[pk]]
                created_at = row[idx.get("created_at")] if "created_at" in idx else None
                text = clean_post(content)
                if not text:
                    skipped["empty"] += 1
                    continue
                if args.skip_seeds and tbl == "post" and text.strip() in seed_contents:
                    skipped["seed"] += 1
                    continue
                if len(text) < args.min_chars:
                    skipped["short"] += 1
                    continue
                if len(text) > args.max_chars:
                    skipped["long"] += 1
                    continue
                meta = user_meta.get(user_id)
                if meta is None or meta["label"] == -1:
                    skipped["missing_label"] += 1
                    continue

                rec = {
                    "synth_post_id": hashlib.sha1(
                        f"oasis|{tbl}|{this_id}|{user_id}|{text[:40]}".encode("utf-8")
                    ).hexdigest()[:16],
                    "text": text,
                    "persona": meta["persona"],
                    "label": meta["label"],
                    "simulation_id": "oasis",
                    "round": 0,
                    "variant": 0,
                    "topic": f"oasis_{tbl}",
                    "n_chars": len(text),
                    "provider": "oasis",
                    "model": "qwen2.5-14b-via-oasis",
                    "oasis_table": tbl,
                    "oasis_row_id": this_id,
                    "oasis_user_id": user_id,
                    "oasis_created_at": created_at,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                if tbl == "post":
                    kept_posts += 1
                else:
                    kept_comments += 1

    print(f"kept total: {kept}  (posts={kept_posts}, comments={kept_comments})")
    print(f"skipped: {skipped}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
