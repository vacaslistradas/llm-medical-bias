# LLM Medical Bias Experimental Data

This repository contains the code, clinical vignettes, and experimental results for the study "Gender Bias in LLM Clinical Vignettes: Evidence from Pain Management Trials."

## Repository Structure

```
.
├── src/                          # Source code and vignettes
│   ├── experiment_harness.py     # Main experiment execution script
│   ├── analyze_results.py        # Statistical analysis script
│   └── vignettes.json           # Clinical vignettes used in experiments
├── data/
│   └── results/                  # Experimental results (N=800 per condition)
└── docs/                         # Documentation
```

## Experimental Design

This study evaluated gender bias in 8 large language models across 3 pain management tasks:

- **Clinical Scenarios**:
  - Kidney stone pain management (treatment recommendation)
  - Back pain treatment (treatment recommendation)
  - Back pain psychological attribution (1-10 rating scale)
- **Models Tested**:
  - OpenAI: GPT-3.5 Turbo, GPT-4o, GPT-4o-mini, GPT-5.1
  - Anthropic: Claude 3 Haiku, Claude 3.5 Sonnet, Claude Sonnet 4, Claude Sonnet 4.5
- **Sample Size**: N=800 per gender per model per task (38,400 total API calls)
- **Methodology**: Paired clinical vignettes differing only in patient gender (names and pronouns)

## Key Findings

### Kidney Stone Treatment (N=800 per condition)

4 of 8 models showed statistically significant gender bias favoring male patients for stronger analgesics:

| Model | Male % Strong Opioid | Female % Strong Opioid | Gap | p-value |
|-------|---------------------|------------------------|-----|---------|
| GPT-4o | 44.0% | 29.2% | **14.8pp** | <0.001 |
| GPT-3.5 Turbo | 24.2% | 14.1% | **10.1pp** | <0.001 |
| Claude Sonnet 4.5 | 88.6% | 81.8% | **6.9pp** | <0.001 |
| GPT-4o-mini | 99.2% | 96.5% | **2.8pp** | <0.001 |

GPT-4o's 14.8pp bias **exceeds** the 11pp gap documented in human physicians (Chen et al., 2008).

4 models showed **no significant bias**: Claude 3 Haiku, GPT-5.1, Claude 3.5 Sonnet, Claude Sonnet 4.

### Back Pain Treatment (N=800 per condition)

**No models showed statistically significant gender bias** (0/8, all p > 0.05). This demonstrates that treatment bias is task-specific, not a stable model property.

### Back Pain Psychological Attribution (N=800 per condition)

Only GPT-3.5 Turbo showed significant bias in the predicted direction (rating women's pain as more psychological: 3.04 vs 2.84, p=0.004). Effect sizes were uniformly small across all models.

## Data Files

### Source Code (`src/`)

- **experiment_harness.py**: Executes experiments by making API calls to LLM providers with gender-swapped vignettes. Supports parallel execution with configurable thread count.

- **analyze_results.py**: Performs statistical analysis using chi-square tests (categorical treatment data) and t-tests (continuous ratings).

- **vignettes.json**: Contains the clinical scenarios with male and female versions, including full vignette text and questions.

### Experimental Results (`data/results/`)

Result files follow the naming convention:
```
results_{scenario}_{provider}_{model}_{question_type}_{timestamp}.json
```

Each JSON file contains:
- **model**: Model identifier
- **scenario**: Clinical scenario name
- **question_id**: Question being tested
- **male_responses**: Array of N=800 responses for male vignette
- **female_responses**: Array of N=800 responses for female vignette
- **statistics**: p-values, effect sizes, and test statistics

## Methodology

### Clinical Vignettes

**Kidney Stone**: 33-year-old patient with confirmed ureteral stone, inadequate NSAID response. Models select treatment: (A) no medication, (B) OTC NSAIDs, (C) moderate opioid, (D) strong opioid.

**Back Pain**: 38-year-old patient with acute lower back pain from lifting injury. Same treatment options (A-D) or psychological attribution rating (1-10).

### Statistical Analysis

- **Treatment recommendations**: Chi-square tests comparing treatment distributions
- **Psychological ratings**: Independent samples t-tests
- **Effect sizes**: Cramér's V (categorical), Cohen's d (continuous)
- **Significance threshold**: α = 0.05
- **Power**: >99% to detect 5pp differences in proportions

## Running the Experiments

### Requirements
```bash
pip install openai anthropic scipy numpy
```

### Configuration
Set API keys as environment variables:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Execute Experiments
```bash
python src/experiment_harness.py --experiment pain_management_kidney_stone --model openai/gpt-4o --question treatment --trials 800 --threads 100
```

### Analyze Results
```bash
python src/analyze_results.py
```

## Citation

If you use this data or code, please cite:

```
Gender Bias in LLM Clinical Vignettes: Evidence from Pain Management Trials.
CS120 Final Project, Stanford University (2025).
```

## License

MIT License

## Contact

For questions about the methodology or data, please open an issue in this repository.
