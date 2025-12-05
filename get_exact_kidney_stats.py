#!/usr/bin/env python3
import json
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

for model_name, filepath in models:
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Parse responses to extract actual answers
    def extract_answer(response):
        import re
        # Try JSON format first
        json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([A-E])"[^{}]*\}', response)
        if json_match:
            return json_match.group(1)
        # Fallback patterns
        for pattern in [r'"answer":\s*"([A-E])"', r'answer["\s:]*([A-E])', r'\{.*?([A-E]).*?\}']:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        return None

    male_answers = [extract_answer(r) for r in data['male_responses']]
    female_answers = [extract_answer(r) for r in data['female_responses']]

    # Count distributions
    male_counts = Counter([a for a in male_answers if a])
    female_counts = Counter([a for a in female_answers if a])

    # Get p-value
    stats = data.get('statistics', {})
    p_value = stats.get('fisher_p_value', stats.get('p_value', 'N/A'))

    print(f"\n{model_name}:")
    print(f"  Male (N={len(male_answers)}):")
    for choice in ['A', 'B', 'C', 'D']:
        count = male_counts.get(choice, 0)
        pct = (count / 50 * 100) if count > 0 else 0
        if count > 0:
            print(f"    {choice}: {count} ({pct:.0f}%)")

    print(f"  Female (N={len(female_answers)}):")
    for choice in ['A', 'B', 'C', 'D']:
        count = female_counts.get(choice, 0)
        pct = (count / 50 * 100) if count > 0 else 0
        if count > 0:
            print(f"    {choice}: {count} ({pct:.0f}%)")

    print(f"  Fisher's exact p = {p_value:.4f}" if isinstance(p_value, float) else f"  Fisher's exact p = {p_value}")
    print(f"  Male modal: {male_counts.most_common(1)[0] if male_counts else 'N/A'}")
    print(f"  Female modal: {female_counts.most_common(1)[0] if female_counts else 'N/A'}")
