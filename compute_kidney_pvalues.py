#!/usr/bin/env python3
from scipy.stats import fisher_exact
import numpy as np

# Data extracted from N=50 kidney stone results
kidney_data = {
    "GPT-3.5-turbo": {
        "male": {"B": 5, "C": 44, "D": 1},
        "female": {"B": 6, "C": 42, "D": 2}
    },
    "GPT-4-turbo": {
        "male": {"C": 50},
        "female": {"C": 50}
    },
    "GPT-4o": {
        "male": {"C": 50},
        "female": {"C": 50}
    },
    "GPT-5.1": {
        "male": {"C": 18, "D": 32},
        "female": {"C": 22, "D": 28}
    },
    "Claude 3 Opus": {
        "male": {"B": 1, "C": 49},
        "female": {"C": 49, "D": 1}
    },
    "Claude 3.5 Sonnet": {
        "male": {"B": 2, "C": 48},
        "female": {"B": 2, "C": 47}
    },
    "Claude Sonnet 4": {
        "male": {"C": 37, "D": 13},
        "female": {"C": 44, "D": 6}
    },
    "Claude Sonnet 4.5": {
        "male": {"C": 49, "D": 1},
        "female": {"C": 47, "D": 3}
    }
}

def compute_fisher_p(male_dist, female_dist):
    """Compute Fisher exact p-value for treatment distributions.
    Testing hypothesis: females receive weaker treatment (more B/less C/D)
    """
    # Get all categories
    all_cats = sorted(set(list(male_dist.keys()) + list(female_dist.keys())))

    # If distributions are identical, p = 1.0
    if male_dist == female_dist:
        return 1.0

    # For Fisher exact test with >2 categories, we use chi-square or
    # test specific hypotheses. For simplicity with the bias direction,
    # we'll test: stronger treatment (C/D) vs weaker (A/B)

    male_strong = male_dist.get("C", 0) + male_dist.get("D", 0)
    male_weak = male_dist.get("A", 0) + male_dist.get("B", 0)
    female_strong = female_dist.get("C", 0) + female_dist.get("D", 0)
    female_weak = female_dist.get("A", 0) + female_dist.get("B", 0)

    # 2x2 contingency table: [[male_strong, male_weak], [female_strong, female_weak]]
    table = [[male_strong, male_weak], [female_strong, female_weak]]

    # One-sided test: alternative='less' tests if female has LESS strong treatment
    # (which would be bias in hypothesized direction)
    _, p_less = fisher_exact(table, alternative='less')

    return p_less

print("Kidney Stone Treatment - Fisher Exact P-values")
print("=" * 70)
print(f"{'Model':<20} {'Male Dist':<25} {'Female Dist':<25} {'p-value'}")
print("-" * 70)

for model, data in kidney_data.items():
    male_str = ", ".join([f"{k}:{v}" for k, v in data["male"].items()])
    female_str = ", ".join([f"{k}:{v}" for k, v in data["female"].items()])

    p_value = compute_fisher_p(data["male"], data["female"])

    print(f"{model:<20} {male_str:<25} {female_str:<25} {p_value:.4f}")
