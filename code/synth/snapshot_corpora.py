"""Generate a Markdown snapshot of every synth corpus produced in this project.

For each corpus we list:
  - file path and absolute size
  - total record count, label distribution, char-length stats
  - 3 random posts per label (so the user can verify content matches label)
  - the AUC from the matching audit_*_summary.csv if available

Output: Assignment 3/results/CORPUS_SNAPSHOTS.md
"""

from __future__ import annotations
from pathlib import Path
import json
import random
import statistics as st

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
SYNTH = A3 / "synth"
RESULTS = A3 / "results"

# (display_name, jsonl_path, audit_tag_for_AUC, n_samples_per_label)
CORPORA = [
    ("Qwen 2.5 14B single-shot (FULL 5k corpus — the one used in main grid)",
     SYNTH / "synth_posts.jsonl", "audit", 3),
    ("Gemini 2.5 Flash — single-shot smoke",
     SYNTH / "smoke_g25flash.jsonl", "audit_g25flash", 2),
    ("Gemini 2.5 Flash-Lite — single-shot smoke",
     SYNTH / "smoke_g25flashlite.jsonl", "audit_g25flashlite", 2),
    ("Gemini 3.1 Flash-Lite — single-shot smoke",
     SYNTH / "smoke_g31flashlite.jsonl", "audit_g31flashlite", 2),
    ("Qwen 2.5 14B + register-roughening preamble — single-shot",
     SYNTH / "smoke_qwen_rough.jsonl", "audit_qwen_rough", 2),
    ("Gemini 2.5 Flash-Lite + register-roughening preamble — single-shot",
     SYNTH / "smoke_g25flashlite_rough.jsonl", "audit_g25flashlite_rough", 2),
    ("OASIS multi-agent — medium run (5 personas mixed actions, posts subset)",
     SYNTH / "oasis_medium_posts.jsonl", "audit_oasis_posts", 2),
    ("OASIS multi-agent — medium run (comments subset)",
     SYNTH / "oasis_medium_comments.jsonl", "audit_oasis_comments", 2),
    ("OASIS multi-agent — POSTS-ONLY at scale (60 agents, CREATE_POST only)",
     SYNTH / "oasis_posts_only.jsonl", "audit_oasis_posts_only", 3),
    ("OASIS POSTS-ONLY — after stripping @mentions, hashtags, post_id back-refs",
     SYNTH / "oasis_posts_stripped.jsonl", "audit_oasis_stripped", 2),
    ("OASIS POSTS-ONLY with Gemini 2.5 Flash-Lite (smoke)",
     SYNTH / "oasis_gemini_smoke.jsonl", "audit_oasis_gemini_smoke", 3),
    ("OASIS POSTS-ONLY with Gemini 2.5 Flash-Lite (scaled, 60 agents, 24 rounds)",
     SYNTH / "oasis_gemini_60.jsonl", "audit_oasis_gemini_60", 3),
    ("Qwen 3.5 9B single-shot (Phase 3 corpus, May 2026 model)",
     SYNTH / "synth_qwen35.jsonl", "audit_qwen35", 3),
    ("DeepSeek-R1 14B single-shot (Phase 3 corpus, Qwen-2 family backbone)",
     SYNTH / "synth_deepseek.jsonl", "audit_deepseek", 3),
    ("Gemini 2.5 Flash-Lite — few-shot real-anchored variant",
     SYNTH / "synth_gemini_fewshot.jsonl", "audit_fewshot_gem", 2),
    ("Claude Opus 4.7 — hand-curated 50-post seed corpus",
     SYNTH / "claude_authored.jsonl", "audit_claude_authored", 5),
]


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def audit_auc(tag):
    p = RESULTS / f"{tag}_summary.csv"
    if not p.exists():
        return None
    line = p.read_text(encoding="utf-8").splitlines()[1].split(",")
    header = p.read_text(encoding="utf-8").splitlines()[0].split(",")
    auc_idx = header.index("discriminator_auc_mean") if "discriminator_auc_mean" in header else None
    if auc_idx is None:
        return None
    try:
        return float(line[auc_idx])
    except Exception:
        return None


def render_corpus(name, path, audit_tag, n_per_label, rng):
    out = [f"## {name}", ""]
    if not path.exists():
        out += ["*(file not present)*", ""]
        return out
    rows = read_jsonl(path)
    if not rows:
        out += ["*(empty corpus)*", ""]
        return out
    by_label = {0: [], 1: []}
    for r in rows:
        by_label.setdefault(r.get("label", -1), []).append(r)
    lens = [r.get("n_chars", len(r["text"])) for r in rows]

    out.append(f"**Path:** `{path.relative_to(A3.parent)}`")
    out.append("")
    auc = audit_auc(audit_tag)
    if auc is not None:
        out.append(f"**Discriminator AUC vs DREDDIT-train:** **{auc:.4f}** (1.0 = trivially separable, 0.5 = indistinguishable)")
        out.append("")
    out.append(f"**Total records:** {len(rows)} · "
               f"**label=1 (stress):** {len(by_label.get(1, []))} · "
               f"**label=0 (non-stress):** {len(by_label.get(0, []))}")
    if lens:
        out.append(f"**Char length:** mean={st.mean(lens):.0f}, median={st.median(lens):.0f}, "
                   f"min={min(lens)}, max={max(lens)}")
    out.append("")
    for lbl in (1, 0):
        pool = by_label.get(lbl, [])
        if not pool:
            continue
        rng.shuffle(pool)
        head = "Stress (label = 1)" if lbl == 1 else "Non-stress (label = 0)"
        out.append(f"### {head}")
        out.append("")
        for rec in pool[:n_per_label]:
            persona = rec.get("persona", "?")
            text = rec["text"].replace("\n", " ").strip()
            snippet = text if len(text) <= 480 else text[:480] + " …"
            out.append(f"- **persona = `{persona}`** — {snippet}")
        out.append("")
    return out


def main():
    rng = random.Random(42)
    lines = ["# Corpus Snapshots", "",
             "Manual quality-check companion for the A3 synthetic corpora. "
             "For each generator below: how many posts, label balance, char-length "
             "summary, the discriminator AUC vs DREDDIT, and a few random posts "
             "per label so you can verify the content matches the persona-derived "
             "label.", "",
             "All synthetic data is consumed by the **training-only** path. The DREDDIT "
             "real test split (n=715, human-labelled) is the *only* evaluation surface.",
             ""]
    for name, path, audit_tag, n in CORPORA:
        lines += render_corpus(name, path, audit_tag, n, rng)
    out_path = RESULTS / "CORPUS_SNAPSHOTS.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
