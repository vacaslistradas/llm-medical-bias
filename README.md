# Non-Monotonic Gender Bias in Medical LLMs (Vignette Study)

This repo contains the materials for a small vignette study (n=30 per gender per model; two clinical scenarios) examining gender differences in pain-management recommendations across eight LLM releases (Dec 2024 runs). It is intended as a transparent supplement to the class paper and a starting point for replication/extension.

## Contents
- `data/` — vignettes, prompts, and parsing/mapping rules.
- `results/` — raw model outputs (as collected), parsed CSVs, and API call logs (timestamps/models/temps) when available.
- `analysis/` — scripts/notebooks for parsing and stats.
- `config/` — run configuration (models, dates, temperatures, trials).
- `paper/` — paper sources (condensed LaTeX/PDF if included).

## Models and runs
- Models: GPT-3.5 Turbo (Nov 2022), GPT-4 Turbo (Apr 2023), GPT-4o (May 2024), GPT-5.1 (Aug 2024), Claude 3 Opus (Mar 2024), Claude 3.5 Sonnet (Jun 2024), Claude Sonnet 4 (Oct 2024), Claude Sonnet 4.5 (Nov 2024).
- Provider: OpenRouter API.
- Dates: 2024-12-02 to 2024-12-04 (single 3-day window).
- Temperature: 0.7; no seed specified.
- Trials: 30 prompts per gender per model per vignette (60/model/condition).
- API version strings were not captured; time stamps are in `results/logs/` if available.

## Reproduction outline
1) Use the exact prompts in `data/prompts.md` and vignettes in `data/vignettes.md`.
2) Call the same models via OpenRouter (or the closest current versions), with temp 0.7.
3) Save raw outputs (JSON/CSV) under `results/raw/` with timestamps; log model name, date, temp, and prompt version in `results/logs/`.
4) Parse with the mapping rules in `data/mapping_rules.md` (scripts in `analysis/scripts/` if provided).
5) Run stats (chi-square/t-tests or exact tests) as in the notebooks/scripts; report counts, p-values, and confidence intervals.

## Notes and limitations
- Two vignettes only (kidney stone analgesia, back pain psych attribution); gender variation via name/pronouns only.
- No prompt variants, temperature sweeps, or repeat-day robustness runs beyond the single window.
- No refusals were observed; ambiguous responses were manually coded per `data/mapping_rules.md`.
- Chi-square assumptions may be weak with small cells; exact tests are preferable for replication.

## License
Specify your preferred license here (e.g., MIT for code, CC-BY 4.0 for text).
