# LLM Medical Gender Bias Study

This repository contains experimental data and analysis scripts for a gender bias study examining clinical vignette responses across eight frontier large language models (December 2025). The study tests for gender-based differences in pain management scenarios where documented physician gender bias exists.

## Key Findings

In kidney stone treatment, **4/8 models showed significant bias** favoring males for stronger analgesics:
- GPT-4o: 14.8pp gap (exceeds documented human physician bias of 11pp)
- GPT-3.5 Turbo: 10.1pp gap
- Claude Sonnet 4.5: 6.9pp gap
- GPT-4o-mini: 2.8pp gap

Four models showed **no significant bias** across any task: GPT-5.1, Claude 3 Haiku, Claude 3.5 Sonnet, and Claude Sonnet 4—outperforming the human physician baseline.

In back pain treatment (identical response options), **no models showed significant bias** (0/8). For psychological attribution, only GPT-3.5 Turbo showed the predicted bias.

## Repository Contents

- `vignettes.json` — Clinical vignettes for two scenarios with male/female variants
- `results_*.json` — Complete experimental results (N=800 per gender per model per task)
- `experiment_harness.py` — Main experimental framework with parallel execution
- `analyze_results.py` — Statistical analysis script
- `compute_kidney_pvalues.py` — Fisher's exact test calculations for kidney stone scenario
- `get_exact_kidney_stats.py` — Detailed kidney stone treatment distribution analysis
- `requirements.txt` — Python dependencies

## Experimental Design

**Models tested** (via OpenRouter API):
- OpenAI: GPT-3.5 Turbo, GPT-4o, GPT-4o-mini, GPT-5.1
- Anthropic: Claude 3 Haiku, Claude 3.5 Sonnet, Claude Sonnet 4, Claude Sonnet 4.5

**Clinical tasks**:
1. Kidney stone pain management (treatment recommendation A-D)
2. Back pain treatment (treatment recommendation A-D)
3. Back pain psychological attribution (1-10 scale)

**Methodology**:
- N=800 trials per gender per model per task
- Total API calls: 38,400 (8 models × 3 tasks × 2 genders × 800 trials)
- Temperature: 0.7
- Gender variation: Patient names and pronouns only
- Data collection: December 2025
- Statistical tests: Chi-square/Fisher's exact (categorical), t-tests (continuous)
- Statistical power: >99% to detect 5pp differences

## Reproduction

1. Install dependencies: `pip install -r requirements.txt`
2. Set up API key: `export OPENROUTER_API_KEY=your_key`
3. Run experiments: `python experiment_harness.py --experiment pain_management_kidney_stone --model openai/gpt-4o --question treatment --trials 800`
4. Analyze results: `python analyze_results.py`

## Limitations

- Two clinical scenarios only; results may not generalize to other domains
- Binary gender markers; no testing of other gender identities
- Gendered names may carry implicit signals beyond gender
- Single time window (December 2025); model behavior may change
- Structured response format may differ from free-form clinical conversations

## License

MIT License - code and data are available for reuse with attribution.
