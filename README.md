# LLM Medical Gender Bias Study

This repository contains experimental data and analysis scripts for a gender bias study examining clinical vignette responses across eight state-of-the-art large language models (December 2025). The study tests for gender-based differences in three clinical scenarios where documented physician gender bias exists.

## Key Findings

Across 24 model-scenario comparisons (8 models × 3 scenarios), no statistically significant gender-based differences were detected in the hypothesized direction (all p > 0.05). However, the sample size (N=50 per gender) provides 80% power to detect only large effects (≥28 percentage points or Cohen's d≥0.57), while physician-level biases documented in the literature are typically 9-10 percentage points—well below the detection threshold of this study.

## Repository Contents

- `vignettes.json` — Clinical vignettes for all three scenarios with male/female variants
- `results_*.json` — Complete experimental results (24 JSON files, N=50 per gender per scenario)
- `experiment_harness.py` — Main experimental framework
- `analyze_results.py` — Statistical analysis script
- `compute_kidney_pvalues.py` — Fisher's exact test calculations for kidney stone scenario
- `get_exact_kidney_stats.py` — Detailed kidney stone treatment distribution analysis
- `extract_kidney_stats.py` — Additional kidney stone data extraction
- `requirements.txt` — Python dependencies
- `.gitignore` — Version control exclusions

## Experimental Design

**Models tested** (via OpenRouter API):
- GPT-3.5 Turbo, GPT-4 Turbo, GPT-4o, GPT-5.1
- Claude 3 Opus, Claude 3.5 Sonnet, Claude Sonnet 4, Claude Sonnet 4.5

**Clinical scenarios**:
1. Kidney stone pain management (treatment recommendation)
2. Chronic back pain (psychological factor attribution)
3. Coronary heart disease (primary diagnosis)

**Methodology**:
- N=50 trials per gender per model per scenario
- Total trials: 2,400 (8 models × 3 scenarios × 2 genders × 50 trials)
- Temperature: 0.7
- Gender variation: Patient names and pronouns only
- Data collection: December 2-4, 2025
- Statistical tests: Fisher's exact test (categorical outcomes), two-sample t-tests (continuous outcomes)

## Reproduction

1. Install dependencies: `pip install -r requirements.txt`
2. Review vignettes: `vignettes.json`
3. Run experiments: `python experiment_harness.py --api-key YOUR_KEY --experiment SCENARIO --model MODEL_NAME --question QUESTION_TYPE --trials 50`
4. Analyze results: `python analyze_results.py` or model-specific scripts

## Limitations

- **Statistical power**: Study can only detect large effects (≥28pp), not physician-magnitude biases (9-10pp)
- **Scenarios**: Three scenarios only; limited coverage of clinical contexts
- **Gender variation**: Name/pronouns only; no other demographic variation
- **Single time window**: December 2-4, 2025; no temporal robustness testing
- **No randomization seeds**: API calls did not specify seeds
- **Limited unparseable responses**: <5 responses out of 2,400 (<0.2%)

## License

MIT License - see code and data are available for reuse with attribution.
