# Code — Assignment 3

**Project:** *When Does LLM-Generated Synthetic Data Augment Reddit Stress Classification? A Regime-Dependent Multi-Generator Study.*
**Subject:** COMP90090 Text Analytics for Health (UniMelb, 2026).
**Student:** Toby Lee (ID 1690649).

## Layout

```
code/
  baseline/    Stage 1 — Twitter-RoBERTa real-only baseline (5 seeds + bootstrap CI).
  synth/       Stage 2 — synthetic-corpus generators (Ollama / Gemini / OASIS / EDA / Claude).
  realism/     Stage 3 — discriminator AUC, vocab Jaccard, class-conditional MMD; UMAP + lift figures.
  grid/        Stage 4 — augmentation grids (Comprehensive U-curve + LLM-tier capability sweep).
  build_docx.py  Render report/report.md → report/NLP4Health_assignment3_LEE.docx.
```

## Environment

- **Hardware:** AMD Ryzen 7 9800X3D, NVIDIA RTX 5070 Ti (16 GB VRAM).
- **OS:** Windows 11.
- **Python:** 3.12 (conda env `uni`). Pinned versions in `code/env_versions.txt`.
- **Local LLM runtime:** Ollama 0.23.1 hosting `qwen2.5:14b-instruct-q4_K_M`, `qwen3.5:9b`, `deepseek-r1:14b`.
- **API keys:** Set `GEMINI_API_KEY` in environment (or in an `Assignment 3/.env.local` file with `GEMINI_API_KEY=...`). Used only by `synth/generate_synth_gemini*.py` and the OASIS+Gemini run in `synth/run_oasis.py`.

## Replication order

End-to-end replication assumes Ollama is running and `GEMINI_API_KEY` is set. Outputs land in `results/` (CSVs), `figures/` (PNGs), and `synth/` (JSONL corpora).

```bash
# 1. Real-only baseline (Stage 1)
python code/baseline/run_baseline.py
#   -> results/baseline_seeds.csv, results/baseline_summary.csv
#   -> outputs/logs/baseline_full.console.log

# 2. Synthetic-corpus generation (Stage 2) — pick the cells you need
python code/synth/generate_synth.py --model qwen2.5:14b-instruct-q4_K_M --target 5000 --out synth/synth_posts.jsonl
python code/synth/generate_synth.py --model qwen3.5:9b              --target 500  --out synth/synth_qwen35.jsonl
python code/synth/generate_synth.py --model deepseek-r1:14b         --target 500  --out synth/synth_deepseek.jsonl
python code/synth/generate_synth_gemini.py --model gemini-2.5-flash       --target 300 --out synth/smoke_g25flash.jsonl
python code/synth/generate_synth_gemini.py --model gemini-2.5-flash-lite  --target 300 --out synth/smoke_g25flashlite.jsonl
python code/synth/generate_synth_gemini.py --model gemini-3.1-flash-lite  --target 300 --out synth/smoke_g31flashlite.jsonl
python code/synth/generate_synth_gemini.py --model gemini-3.5-flash       --target 500 --out synth/synth_gemini35flash.jsonl
python code/synth/generate_synth_gemini_fewshot.py                        --target 500 --out synth/synth_gemini_fewshot.jsonl
python code/synth/run_oasis.py            --backend qwen   --out synth/oasis_posts_only.jsonl
python code/synth/run_oasis.py            --backend gemini --out synth/oasis_gemini_60.jsonl
python code/synth/claude_authored_seed.py # writes synth/claude_authored.jsonl

# 3. Distributional realism audit (Stage 3) — per synthetic corpus
python code/realism/audit.py --synth synth/synth_posts.jsonl     --tag audit_qwen300
python code/realism/audit.py --synth synth/synth_qwen35.jsonl    --tag audit_qwen35
python code/realism/audit.py --synth synth/synth_deepseek.jsonl  --tag audit_deepseek
# (repeat for each generator; outputs results/audit_<tag>_summary.csv and figures/<tag>_umap.png)

# Optional: embedding-filter the Qwen 2.5 corpus into a top-1000 subset
python code/realism/filter_synth_by_realism.py # writes synth/synth_filtered_top1000.jsonl

# 4. Augmentation grids (Stage 4)
python code/grid/run_learning_curve.py            # results/learning_curve.csv
python code/grid/run_comprehensive_grid.py        # results/comprehensive_grid_seeds.csv (175 trainings)
python code/grid/run_llm_tier_grid.py             # results/llm_tier_grid_seeds.csv     (130 trainings)

# 5. Figures
python code/realism/plot_u_curve.py               # figures/augmentation_u_curve.png   (report Figure 1)
python code/realism/plot_capability_vs_lift.py    # figures/auc_vs_lift.png            (report Figure 2)

# 6. Render report
python code/synth/snapshot_corpora.py             # results/CORPUS_SNAPSHOTS.md
python code/make_report_data.py                   # results/auto_report_tables.md
python code/build_docx.py                         # report/NLP4Health_assignment3_LEE.docx
```

## AI coding-assistant disclosure

I used Anthropic Claude (Opus 4.7) extensively during this project. The assistant scaffolded the synthetic-data generators, the distributional realism audit, the augmentation-grid runners, and the report-rendering pipeline; I read, edited, tested, and curated every component. The Claude-authored synthetic corpus in `synth/claude_authored.jsonl` was written directly by the assistant under my direction as one of the experimental conditions (a *test variable*, disclosed in §2.2 and Table 1 of the report). The assistant also produced an initial prose draft of the report against the numerical results, which I substantially revised. All technical decisions — research question, choice of classifier, persona inventory, grid composition, regime-dependent framing — are mine. Use of the assistant is consistent with the project rubric's encouragement of library use and with the lecturer's pre-commit ruling of 7 May 2026.
