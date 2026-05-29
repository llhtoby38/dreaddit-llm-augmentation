"""Read a JSONL of Claude-authored posts and validate / load them.

The Claude-authored posts in `synth/claude_authored.jsonl` are written directly
by Claude (with full context of DREDDIT register and the project's task) as
a comparison data point against the LLM-only generators. This is a tiny
N (~100-200) high-effort, high-context corpus.

Records have the same schema as the other synth corpora.
"""
from __future__ import annotations
from pathlib import Path
import json
import sys

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
JSONL = A3 / "synth" / "claude_authored.jsonl"


def main():
    if not JSONL.exists():
        sys.exit(f"missing {JSONL} — author the posts there first")
    rows = []
    with JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            assert {"text", "label", "persona"} <= set(r.keys()), r
            rows.append(r)
    print(f"loaded {len(rows)} claude-authored posts")
    by_label = {0: 0, 1: 0}
    by_persona = {}
    char_lens = []
    for r in rows:
        by_label[r["label"]] += 1
        by_persona[r["persona"]] = by_persona.get(r["persona"], 0) + 1
        char_lens.append(len(r["text"]))
    print(f"label balance: {by_label}")
    print(f"persona spread ({len(by_persona)} personas):")
    for p, n in sorted(by_persona.items(), key=lambda x: -x[1]):
        print(f"  {p:25s} {n}")
    import statistics as st
    print(f"char length: mean={st.mean(char_lens):.0f} median={st.median(char_lens):.0f} min={min(char_lens)} max={max(char_lens)}")


if __name__ == "__main__":
    main()
