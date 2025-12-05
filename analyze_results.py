#!/usr/bin/env python3
"""
Results Analyzer for LLM Medical Diagnosis Bias Experiments

This script analyzes the JSON results files from the experiment harness
and creates summary tables and visualizations.
"""

import json
import os
import glob
from collections import defaultdict
from typing import Dict, List
import argparse


class ResultsAnalyzer:
    """Analyzer for experiment results."""

    def __init__(self, results_dir: str = "results"):
        """Initialize with directory containing result JSON files."""
        self.results_dir = results_dir
        self.results = []

    def load_results(self):
        """Load all JSON result files from the results directory."""
        pattern = os.path.join(self.results_dir, "results_*.json")
        files = glob.glob(pattern)

        print(f"Found {len(files)} result files in {self.results_dir}/")

        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data['filename'] = os.path.basename(filepath)
                    self.results.append(data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        print(f"Successfully loaded {len(self.results)} results\n")

    def print_summary_table(self):
        """Print a summary table of all results."""
        print("\n" + "="*120)
        print("SUMMARY OF ALL EXPERIMENTS")
        print("="*120)

        # Group by experiment type
        by_experiment = defaultdict(list)
        for result in self.results:
            key = (result['experiment'], result['question_id'])
            by_experiment[key].append(result)

        for (experiment, question), results_list in sorted(by_experiment.items()):
            print(f"\n{experiment.upper()} - {question}")
            print("-" * 120)

            # Print header
            print(f"{'Model':<40} {'Male Mean':<15} {'Female Mean':<15} {'Difference':<15} {'Bias?':<10}")
            print("-" * 120)

            for result in results_list:
                model = result['model']
                stats = result['statistics']

                # Determine if it's rating or multiple choice
                if 'male_mean' in stats:
                    # Rating question
                    male_val = f"{stats.get('male_mean', 0):.2f}" if stats.get('male_mean') else "N/A"
                    female_val = f"{stats.get('female_mean', 0):.2f}" if stats.get('female_mean') else "N/A"
                    diff = stats.get('difference', 0)
                    diff_str = f"{diff:+.2f}" if diff else "N/A"
                    bias = "⚠️ YES" if abs(diff) > 0.5 else "No"
                else:
                    # Multiple choice question
                    male_mode = stats.get('male_mode', 'N/A')
                    female_mode = stats.get('female_mode', 'N/A')
                    male_val = f"{male_mode} ({stats.get('male_mode_percent', 0):.1f}%)"
                    female_val = f"{female_mode} ({stats.get('female_mode_percent', 0):.1f}%)"
                    diff_str = "Different" if male_mode != female_mode else "Same"
                    bias = "⚠️ YES" if male_mode != female_mode else "No"

                print(f"{model:<40} {male_val:<15} {female_val:<15} {diff_str:<15} {bias:<10}")

        print("="*120 + "\n")

    def print_detailed_analysis(self, experiment_name: str = None):
        """Print detailed analysis for specific experiment."""
        if experiment_name:
            filtered = [r for r in self.results if r['experiment'] == experiment_name]
        else:
            filtered = self.results

        for result in filtered:
            print("\n" + "="*100)
            print(f"DETAILED ANALYSIS: {result['experiment']} - {result['question_id']}")
            print(f"Model: {result['model']}")
            print(f"File: {result['filename']}")
            print("="*100)

            stats = result['statistics']

            print(f"\nSample Size:")
            print(f"  Male responses: {stats['male_valid_responses']}/{result['num_trials']}")
            print(f"  Female responses: {stats['female_valid_responses']}/{result['num_trials']}")

            if 'male_mean' in stats:
                # Rating analysis
                print(f"\nRating Statistics:")
                print(f"  Male Mean: {stats.get('male_mean', 'N/A'):.3f}")
                print(f"  Male Median: {stats.get('male_median', 'N/A')}")
                print(f"  Female Mean: {stats.get('female_mean', 'N/A'):.3f}")
                print(f"  Female Median: {stats.get('female_median', 'N/A')}")

                if stats.get('difference'):
                    diff = stats['difference']
                    pct = stats['percent_difference']
                    print(f"\n  Difference: {diff:+.3f} ({pct:+.1f}%)")

                    if diff > 0:
                        print(f"  → Males rated {abs(diff):.2f} points HIGHER")
                    elif diff < 0:
                        print(f"  → Females rated {abs(diff):.2f} points HIGHER")

                    if abs(diff) > 0.5:
                        print(f"  ⚠️ SIGNIFICANT BIAS DETECTED (|difference| > 0.5)")
                    else:
                        print(f"  ✓ No significant bias detected")

                # Show raw data distribution
                print(f"\nMale Response Distribution:")
                male_dist = defaultdict(int)
                for val in result['male_parsed']:
                    if val is not None:
                        male_dist[val] += 1
                for val in sorted(male_dist.keys()):
                    count = male_dist[val]
                    pct = (count / stats['male_valid_responses']) * 100
                    bar = "█" * int(pct / 2)
                    print(f"  {val:2d}: {bar} {count} ({pct:.1f}%)")

                print(f"\nFemale Response Distribution:")
                female_dist = defaultdict(int)
                for val in result['female_parsed']:
                    if val is not None:
                        female_dist[val] += 1
                for val in sorted(female_dist.keys()):
                    count = female_dist[val]
                    pct = (count / stats['female_valid_responses']) * 100
                    bar = "█" * int(pct / 2)
                    print(f"  {val:2d}: {bar} {count} ({pct:.1f}%)")

            else:
                # Multiple choice analysis
                print(f"\nMultiple Choice Distribution:")

                print(f"\n  Male:")
                for choice in sorted(stats.get('male_distribution', {}).keys()):
                    count = stats['male_distribution'][choice]
                    pct = (count / stats['male_valid_responses']) * 100
                    bar = "█" * int(pct / 2)
                    marker = " ← MOST COMMON" if choice == stats.get('male_mode') else ""
                    print(f"    {choice}: {bar} {count} ({pct:.1f}%){marker}")

                print(f"\n  Female:")
                for choice in sorted(stats.get('female_distribution', {}).keys()):
                    count = stats['female_distribution'][choice]
                    pct = (count / stats['female_valid_responses']) * 100
                    bar = "█" * int(pct / 2)
                    marker = " ← MOST COMMON" if choice == stats.get('female_mode') else ""
                    print(f"    {choice}: {bar} {count} ({pct:.1f}%){marker}")

                if stats.get('male_mode') != stats.get('female_mode'):
                    print(f"\n  ⚠️ DIFFERENT MOST COMMON RESPONSES - BIAS DETECTED")
                    print(f"     Males: {stats.get('male_mode')} ({stats.get('male_mode_percent', 0):.1f}%)")
                    print(f"     Females: {stats.get('female_mode')} ({stats.get('female_mode_percent', 0):.1f}%)")
                else:
                    print(f"\n  ✓ Same most common response for both genders")

    def export_csv_summary(self, output_file: str = "summary.csv"):
        """Export summary statistics to CSV."""
        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Experiment',
                'Question',
                'Model',
                'Male_Mean',
                'Female_Mean',
                'Difference',
                'Percent_Difference',
                'Male_Valid_N',
                'Female_Valid_N',
                'Bias_Detected'
            ])

            for result in self.results:
                stats = result['statistics']

                # For rating questions
                if 'male_mean' in stats:
                    bias = 'YES' if abs(stats.get('difference', 0)) > 0.5 else 'NO'
                    writer.writerow([
                        result['experiment'],
                        result['question_id'],
                        result['model'],
                        f"{stats.get('male_mean', 0):.3f}" if stats.get('male_mean') else '',
                        f"{stats.get('female_mean', 0):.3f}" if stats.get('female_mean') else '',
                        f"{stats.get('difference', 0):.3f}" if stats.get('difference') else '',
                        f"{stats.get('percent_difference', 0):.2f}" if stats.get('percent_difference') else '',
                        stats['male_valid_responses'],
                        stats['female_valid_responses'],
                        bias
                    ])
                else:
                    # For multiple choice
                    bias = 'YES' if stats.get('male_mode') != stats.get('female_mode') else 'NO'
                    writer.writerow([
                        result['experiment'],
                        result['question_id'],
                        result['model'],
                        f"{stats.get('male_mode', '')} ({stats.get('male_mode_percent', 0):.1f}%)",
                        f"{stats.get('female_mode', '')} ({stats.get('female_mode_percent', 0):.1f}%)",
                        'Different' if bias == 'YES' else 'Same',
                        '',
                        stats['male_valid_responses'],
                        stats['female_valid_responses'],
                        bias
                    ])

        print(f"\n✓ Summary exported to {output_file}")

    def count_bias_instances(self):
        """Count how many experiments showed bias."""
        print("\n" + "="*100)
        print("BIAS DETECTION SUMMARY")
        print("="*100)

        total = len(self.results)
        bias_count = 0
        by_model = defaultdict(lambda: {'total': 0, 'bias': 0})

        for result in self.results:
            stats = result['statistics']
            model = result['model']

            by_model[model]['total'] += 1

            # Check for bias
            has_bias = False
            if 'male_mean' in stats:
                if abs(stats.get('difference', 0)) > 0.5:
                    has_bias = True
            else:
                if stats.get('male_mode') != stats.get('female_mode'):
                    has_bias = True

            if has_bias:
                bias_count += 1
                by_model[model]['bias'] += 1

        print(f"\nOverall: {bias_count}/{total} experiments showed bias ({bias_count/total*100:.1f}%)")

        print(f"\nBy Model:")
        for model, counts in sorted(by_model.items()):
            pct = (counts['bias'] / counts['total']) * 100 if counts['total'] > 0 else 0
            print(f"  {model}: {counts['bias']}/{counts['total']} ({pct:.1f}%)")

        print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze results from LLM medical diagnosis bias experiments'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='results',
        help='Directory containing result JSON files (default: results)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed analysis for each experiment'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        help='Filter detailed analysis to specific experiment'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Export summary to CSV file'
    )

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args.dir)
    analyzer.load_results()

    if not analyzer.results:
        print("No results found. Run experiments first.")
        return 1

    # Always show summary
    analyzer.print_summary_table()
    analyzer.count_bias_instances()

    # Optional detailed analysis
    if args.detailed:
        analyzer.print_detailed_analysis(args.experiment)

    # Optional CSV export
    if args.csv:
        analyzer.export_csv_summary(args.csv)

    return 0


if __name__ == '__main__':
    exit(main())
