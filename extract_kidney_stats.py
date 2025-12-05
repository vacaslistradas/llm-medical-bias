#!/usr/bin/env python3
import json
import os
from collections import Counter

# Model files
models = [
    ("GPT-3.5-turbo", "github-repo/data/validated_n50_results/results_pain_management_kidney_stone_openai_gpt-3.5-turbo_treatment_20251204_170346.json"),
    ("GPT-4-turbo", "github-repo/data/validated_n50_results/results_pain_management_kidney_stone_openai_gpt-4-turbo_treatment_20251204_171555.json"),
    ("GPT-4o", "github-repo/data/validated_n50_results/results_pain_management_kidney_stone_openai_gpt-4o_treatment_20251204_170825.json"),
    ("GPT-5.1", "github-repo/data/validated_n50_results/results_pain_management_kidney_stone_openai_gpt-5.1_treatment_20251204_171602.json"),
    ("Claude 3 Opus", "github-repo/data/validated_n50_results/results_pain_management_kidney_stone_anthropic_claude-3-opus_treatment_20251204_171937.json"),
    ("Claude 3.5 Sonnet", "github-repo/data/validated_n50_results/results_pain_management_kidney_stone_anthropic_claude-3.5-sonnet_treatment_20251204_171605.json"),
    ("Claude Sonnet 4", "github-repo/data/validated_n50_results/results_pain_management_kidney_stone_anthropic_claude-sonnet-4_treatment_20251204_171653.json"),
    ("Claude Sonnet 4.5", "github-repo/data/validated_n50_results/results_pain_management_kidney_stone_anthropic_claude-sonnet-4.5_treatment_20251204_171935.json"),
]

print("Kidney Stone Treatment Results (N=50 per gender)\n")
print(f"{'Model':<20} {'Male Distribution':<30} {'Female Distribution':<30} {'Fisher p':<10}")
print("=" * 90)

for model_name, filepath in models:
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Get responses
    male_resp = data['male_responses']
    female_resp = data['female_responses']

    # Count distributions
    male_counts = Counter(male_resp)
    female_counts = Counter(female_resp)

    # Format distributions
    male_dist = ', '.join([f"{k}:{male_counts[k]}" for k in sorted(male_counts.keys())])
    female_dist = ', '.join([f"{k}:{female_counts[k]}" for k in sorted(female_counts.keys())])

    # Get p-value
    p_value = data.get('statistics', {}).get('fisher_p_value', 'N/A')
    if isinstance(p_value, float):
        p_str = f"{p_value:.3f}"
    else:
        p_str = str(p_value)

    print(f"{model_name:<20} {male_dist:<30} {female_dist:<30} {p_str:<10}")

    # More detailed breakdown
    print(f"  Male N={len(male_resp)}, Female N={len(female_resp)}")
    print(f"  Male: {dict(male_counts)}")
    print(f"  Female: {dict(female_counts)}")
    print()
