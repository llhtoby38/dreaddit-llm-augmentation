# When Does LLM-Generated Synthetic Data Augment Reddit Stress Classification?

A regime-dependent multi-generator study of LLM-driven training-data augmentation
for binary stress classification on the [Dreaddit](https://www.cs.columbia.edu/~elsbeth/dreaddit.html)
corpus (Turcan & McKeown, EMNLP-LOUHI 2019).

Independent project for COMP90090 *Text Analytics for Health* (University of Melbourne).

## Headline findings

- **Augmentation lift is regime-dependent.** At 100 real training posts every
  augmentation cell beats the real-only baseline (best lift Δ = +0.083 macro-F1);
  by 500 real posts every cell matches or hurts; at the full 2,838 corpus every
  cell sits within ±0.005 of baseline.
- **Discriminator AUC and downstream utility are decoupled** across 11 generators.
  Pearson r ≈ 0 between AUC and lift (r = −0.19 at n=100, +0.18 at n=250). A
  generator that is easier to distinguish from real Dreaddit is not less useful
  for augmentation.
- **Source diversity > single-source scaling.** Mixing 100 Qwen 2.5 + 100 Gemini
  3.1 + 50 Claude posts beats any single-source cell. Multi-agent simulation
  (OASIS) ranks second when paired with a frontier API backbone.
- **Within-family capability monotonicity** (Qwen 2.5 < DeepSeek-R1 ≈ Qwen 3.5)
  coexists with **across-family non-monotonicity** (older Gemini 2.5 Flash >
  newer Gemini 3.5 Flash) — newest ≠ most useful.

## Repository layout

```
code/
  baseline/      Stage 1 — Twitter-RoBERTa real-only baseline (5 seeds + bootstrap CI).
  synth/         Stage 2 — synthetic-corpus generators (Ollama, Gemini, OASIS, EDA, Claude).
  realism/       Stage 3 — discriminator AUC, vocab Jaccard, class-conditional MMD; UMAP + lift figures.
  grid/          Stage 4 — augmentation grids (Comprehensive U-curve + LLM-tier capability sweep).
  build_docx.py  Render the report from markdown.
  README.md      Detailed replication order (12 numbered steps).
```

See **`code/README.md`** for the full replication walk-through, environment requirements, and AI coding-assistant disclosure.

## What this repo does NOT contain

- **Synthetic corpora.** The mental-health-adjacent synthetic posts are not
  redistributed (dual-use risk for harassment-bot training or misinformation,
  per the report's Ethics Statement). The generation scripts are included so
  any consumer can produce the corpora locally.
- **Raw real data.** Dreaddit is publicly available from the corpus's official
  page; download separately and point the scripts at the resulting CSV.
- **API keys.** Set `GEMINI_API_KEY` in your environment to run the Gemini
  generators. Local-LLM generators only need a running Ollama daemon.

## Citation

```
@misc{lee2026dreaddit_llm_aug,
  author       = {Toby Lee},
  title        = {When Does LLM-Generated Synthetic Data Augment Reddit Stress Classification? A Regime-Dependent Multi-Generator Study},
  year         = {2026},
  howpublished = {Independent project, COMP90090 Text Analytics for Health, University of Melbourne},
  url          = {https://github.com/llhtoby38/dreaddit-llm-augmentation}
}
```

## License

MIT (see [`LICENSE`](LICENSE)).
