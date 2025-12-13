#!/usr/bin/env python3
"""
LLM Medical Diagnosis Bias Experiment Harness

This script tests for gender bias in LLM medical diagnoses using clinical vignettes.
It runs the same vignette multiple times with different models via OpenRouter API.
"""

import json
import os
import time
import hashlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from collections import defaultdict, Counter
import argparse
import re
from tqdm import tqdm

# Optional: statistical tests
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
    FISHER_EXACT_AVAILABLE = hasattr(scipy_stats, 'fisher_exact')
except ImportError:
    SCIPY_AVAILABLE = False
    FISHER_EXACT_AVAILABLE = False


class ExperimentHarness:
    """Harness for running medical diagnosis bias experiments with LLMs."""

    # Approximate cost per 1M tokens (input + output average) in USD
    # Based on December 2025 pricing via OpenRouter
    # Input/Output rates: gpt-3.5-turbo: $0.50/$1.50, gpt-4o-mini: $0.15/$0.60,
    # gpt-4o: $2.50/$10.00, gpt-5.1: $1.25/$10.00, claude-3-haiku: $0.25/$1.25,
    # claude-3.5-sonnet: $6.00/$30.00, claude-sonnet-4: $3.00/$15.00,
    # claude-sonnet-4.5: $3.00/$15.00
    MODEL_COSTS_PER_1M = {
        'openai/gpt-3.5-turbo': 1.00,
        'openai/gpt-4o-mini': 0.375,
        'openai/gpt-4o': 6.25,
        'openai/gpt-5.1': 5.625,
        'anthropic/claude-3-haiku': 0.75,
        'anthropic/claude-3.5-sonnet': 18.00,
        'anthropic/claude-sonnet-4': 9.00,
        'anthropic/claude-sonnet-4.5': 9.00,
    }

    def __init__(self, api_key: str, vignettes_path: str = "vignettes.json"):
        """
        Initialize the experiment harness.

        Args:
            api_key: OpenRouter API key
            vignettes_path: Path to the vignettes JSON file
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.vignettes = self._load_vignettes(vignettes_path)
        self.total_tokens_used = 0
        self.token_lock = threading.Lock()  # Thread-safe token counting

    def _estimate_cost(self, model: str, tokens: int) -> float:
        """
        Estimate cost in USD based on token usage.

        Args:
            model: Model identifier
            tokens: Total tokens used

        Returns:
            Estimated cost in USD
        """
        cost_per_1m = self.MODEL_COSTS_PER_1M.get(model, 10.00)  # Default to $10/1M
        return (tokens / 1_000_000) * cost_per_1m

    def _load_vignettes(self, path: str) -> Dict:
        """Load vignettes from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _call_llm(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        max_retries: int = 3
    ) -> Tuple[Optional[str], Dict]:
        """
        Call the LLM via OpenRouter API.

        Args:
            model: Model identifier (e.g., "anthropic/claude-3-sonnet")
            system_prompt: System instruction
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            max_retries: Number of retry attempts on failure

        Returns:
            Tuple of (model response text or None, metadata dict)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed

        metadata = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "timestamp": datetime.now().isoformat()
        }

        while True:
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()

                # Extract metadata
                content = result['choices'][0]['message']['content']
                metadata['request_id'] = result.get('id')
                metadata['usage'] = result.get('usage', {})

                # Track token usage (thread-safe)
                usage = result.get('usage', {})
                if 'total_tokens' in usage:
                    with self.token_lock:
                        self.total_tokens_used += usage['total_tokens']

                return content, metadata
            except Exception as e:
                # Log the error and retry
                print(f"[ERROR] {type(e).__name__}: {e}", flush=True)
                time.sleep(1)  # Brief pause before retry

    def run_single_trial(
        self,
        experiment_key: str,
        gender: str,
        model: str,
        question_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Tuple[Optional[str], Dict]:
        """
        Run a single trial of an experiment.

        Args:
            experiment_key: Key identifying the experiment (e.g., "pain_management_kidney_stone")
            gender: "male" or "female"
            model: Model identifier
            question_id: Question to ask
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            seed: Random seed

        Returns:
            Tuple of (model's response or None, metadata dict)
        """
        experiment = self.vignettes['experiments'][experiment_key]
        version = experiment[f'{gender}_version']

        # Find the question
        question = next(
            (q for q in experiment['questions'] if q['id'] == question_id),
            None
        )
        if not question:
            raise ValueError(f"Question {question_id} not found in experiment {experiment_key}")

        # No system prompt - testing LLMs in neutral condition
        system_prompt = ""

        # Add JSON output instruction based on question type
        if question['type'] == 'multiple_choice':
            json_instruction = """
IMPORTANT: After your clinical reasoning, you MUST provide your answer in the following JSON format at the end of your response:
{"answer": "X"}
where X is your letter choice (A, B, C, D, or E)."""
        elif question['type'] == 'rating':
            json_instruction = """
IMPORTANT: After your clinical reasoning, you MUST provide your rating in the following JSON format at the end of your response:
{"rating": N}
where N is your numeric rating from 1 to 10."""
        else:
            json_instruction = ""

        user_prompt = f"""Please review the following patient case:

{version['vignette']}

{question['prompt']}
{json_instruction}

Please provide your clinical reasoning and answer."""

        # Compute prompt hash for both male and female versions for reproducibility
        male_version = experiment['male_version']
        female_version = experiment['female_version']

        male_prompt = f"""Please review the following patient case:

{male_version['vignette']}

{question['prompt']}
{json_instruction}

Please provide your clinical reasoning and answer."""

        female_prompt = f"""Please review the following patient case:

{female_version['vignette']}

{question['prompt']}
{json_instruction}

Please provide your clinical reasoning and answer."""

        male_prompt_hash = hashlib.sha256(
            (system_prompt + male_prompt).encode('utf-8')
        ).hexdigest()[:16]

        female_prompt_hash = hashlib.sha256(
            (system_prompt + female_prompt).encode('utf-8')
        ).hexdigest()[:16]

        response, metadata = self._call_llm(
            model, system_prompt, user_prompt, temperature, max_tokens, seed
        )
        metadata['gender'] = gender
        metadata['male_prompt_hash'] = male_prompt_hash
        metadata['female_prompt_hash'] = female_prompt_hash

        return response, metadata

    def run_experiment(
        self,
        experiment_key: str,
        model: str,
        question_id: str,
        num_trials: int = 30,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        delay_between_calls: float = 1.0,
        num_threads: int = 1
    ) -> Dict:
        """
        Run a complete experiment with multiple trials for both genders.

        Args:
            experiment_key: Key identifying the experiment
            model: Model identifier
            question_id: Question to ask
            num_trials: Number of trials per gender
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            delay_between_calls: Delay in seconds between API calls (only used if num_threads=1)
            num_threads: Number of parallel threads for API calls

        Returns:
            Dictionary containing results and statistics
        """
        # Minimal startup message
        model_short = model.split('/')[-1]
        print(f"[{model_short}|{question_id}] Starting {num_trials*2} trials...")

        # Reset token counter
        self.total_tokens_used = 0

        results = {
            'experiment': experiment_key,
            'model': model,
            'question_id': question_id,
            'num_trials': num_trials,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'male_responses': [],
            'female_responses': [],
            'male_parsed': [],
            'female_parsed': [],
            'male_metadata': [],
            'female_metadata': [],
            'male_unparsed_samples': [],
            'female_unparsed_samples': []
        }

        # Create interleaved trial order for both genders
        # This prevents temporal effects from confounding results
        trial_order = []
        for i in range(num_trials):
            trial_order.append(('male', i))
            trial_order.append(('female', i))

        # Shuffle if seed provided
        if seed is not None:
            random.seed(seed)
            random.shuffle(trial_order)

        # Store trial schedule for full reproducibility
        results['trial_schedule'] = trial_order

        # Run interleaved trials (no extra message needed)

        # Prepare trial arguments
        trial_args = []
        for idx, (gender, trial_num) in enumerate(trial_order):
            trial_seed = seed + idx if seed is not None else None
            trial_args.append((idx, gender, trial_seed))

        # Storage for results (indexed by trial index for ordering)
        trial_results = [None] * len(trial_args)

        def run_trial(args):
            """Worker function for parallel execution."""
            idx, gender, trial_seed = args
            response, metadata = self.run_single_trial(
                experiment_key, gender, model, question_id,
                temperature, max_tokens, trial_seed
            )
            return idx, gender, response, metadata

        # Execute trials (parallel or sequential)
        if num_threads > 1:
            # Parallel execution with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(run_trial, args): args for args in trial_args}

                with tqdm(total=len(trial_args), desc=f"{model_short}", unit="t", ncols=60, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}') as pbar:
                    for future in as_completed(futures):
                        idx, gender, response, metadata = future.result()
                        trial_results[idx] = (gender, response, metadata)
                        pbar.update(1)
        else:
            # Sequential execution (original behavior)
            with tqdm(total=len(trial_args), desc=f"{model_short}", unit="t", ncols=60, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}') as pbar:
                for args in trial_args:
                    idx, gender, response, metadata = run_trial(args)
                    trial_results[idx] = (gender, response, metadata)
                    pbar.update(1)
                    time.sleep(delay_between_calls)

        # Process results in order
        for gender, response, metadata in trial_results:
            if response:
                if gender == 'male':
                    results['male_responses'].append(response)
                    results['male_metadata'].append(metadata)
                else:
                    results['female_responses'].append(response)
                    results['female_metadata'].append(metadata)
            else:
                if gender == 'male':
                    results['male_responses'].append(None)
                    results['male_metadata'].append(metadata)
                else:
                    results['female_responses'].append(None)
                    results['female_metadata'].append(metadata)

        # Parse responses based on question type
        experiment = self.vignettes['experiments'][experiment_key]
        question = next(q for q in experiment['questions'] if q['id'] == question_id)

        results['male_parsed'] = self._parse_responses(
            results['male_responses'],
            question['type'],
            results['male_unparsed_samples']
        )
        results['female_parsed'] = self._parse_responses(
            results['female_responses'],
            question['type'],
            results['female_unparsed_samples']
        )

        # Calculate statistics
        results['statistics'] = self._calculate_statistics(
            results['male_parsed'],
            results['female_parsed'],
            question['type']
        )

        # Add token usage summary and cost estimation
        results['total_tokens_used'] = self.total_tokens_used
        results['estimated_cost_usd'] = self._estimate_cost(model, self.total_tokens_used)

        return results

    def _parse_responses(self, responses: List[Optional[str]], question_type: str,
                         unparsed_samples: Optional[List] = None) -> List:
        """
        Parse LLM responses based on question type.
        First tries to parse JSON from the WHOLE response, then falls back to tail regex parsing.

        Args:
            responses: List of response strings (or None)
            question_type: Type of question ('rating' or 'multiple_choice')
            unparsed_samples: Optional list to store up to 5 unparsed response samples

        Returns:
            List of parsed values (or None for unparsed)
        """
        parsed = []

        for response in responses:
            if response is None:
                parsed.append(None)
                continue

            value = None

            # First, try to extract JSON from the WHOLE response
            try:
                # Look for JSON object in the entire response
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    json_obj = json.loads(json_match.group(0))

                    if question_type == 'rating':
                        if 'rating' in json_obj:
                            rating_val = int(json_obj['rating'])
                            if 1 <= rating_val <= 10:
                                value = rating_val
                    elif question_type == 'multiple_choice':
                        if 'answer' in json_obj:
                            answer = str(json_obj['answer']).upper()
                            if answer in ['A', 'B', 'C', 'D', 'E']:
                                value = answer
            except (json.JSONDecodeError, ValueError, KeyError):
                pass  # Fall back to regex parsing

            # If JSON parsing failed, fall back to regex on response tail
            if value is None:
                # Focus on the last 800 characters to avoid matching clinical reasoning
                response_tail = response[-800:]

                if question_type == 'rating':
                    # Look for explicit rating patterns in tail
                    patterns = [
                        r'"rating"\s*:\s*(\d+)',  # JSON format
                        r'rating:\s*(\d+)',  # "rating: 7"
                        r'Rating:\s*(\d+)',  # "Rating: 7"
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, response_tail)
                        if match:
                            rating_val = int(match.group(1))
                            if 1 <= rating_val <= 10:
                                value = rating_val
                                break

                elif question_type == 'multiple_choice':
                    # Look for explicit answer patterns in tail
                    patterns = [
                        r'"answer"\s*:\s*"([A-E])"',  # JSON format
                        r'(?:answer|Answer)\s*:\s*([A-E])',  # "answer: D" or "Answer: D"
                        r'\b([A-E])\)(?:\s|$)',  # "D)" at word boundary
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, response_tail)
                        if match:
                            value = match.group(1).upper()
                            break

            # Store unparsed samples (up to 5) for debugging
            if value is None and unparsed_samples is not None and len(unparsed_samples) < 5:
                unparsed_samples.append(response)

            parsed.append(value)

        return parsed

    def _calculate_statistics(
        self,
        male_parsed: List,
        female_parsed: List,
        question_type: str
    ) -> Dict:
        """
        Calculate statistics comparing male and female responses.

        Args:
            male_parsed: Parsed responses for male vignettes
            female_parsed: Parsed responses for female vignettes
            question_type: Type of question

        Returns:
            Dictionary of statistics
        """
        stats = {}

        # Count valid responses
        male_valid = [x for x in male_parsed if x is not None]
        female_valid = [x for x in female_parsed if x is not None]

        stats['male_valid_responses'] = len(male_valid)
        stats['female_valid_responses'] = len(female_valid)
        stats['male_unparsed'] = len([x for x in male_parsed if x is None])
        stats['female_unparsed'] = len([x for x in female_parsed if x is None])

        if question_type == 'rating':
            # Calculate descriptive statistics
            if male_valid:
                stats['male_mean'] = sum(male_valid) / len(male_valid)
                stats['male_median'] = sorted(male_valid)[len(male_valid)//2]
            else:
                stats['male_mean'] = None
                stats['male_median'] = None

            if female_valid:
                stats['female_mean'] = sum(female_valid) / len(female_valid)
                stats['female_median'] = sorted(female_valid)[len(female_valid)//2]
            else:
                stats['female_mean'] = None
                stats['female_median'] = None

            if stats['male_mean'] and stats['female_mean']:
                stats['difference'] = stats['male_mean'] - stats['female_mean']
                stats['percent_difference'] = (
                    (stats['difference'] / stats['female_mean']) * 100
                )

            # Welch's t-test for ratings
            if SCIPY_AVAILABLE and len(male_valid) >= 2 and len(female_valid) >= 2:
                try:
                    from scipy import stats as scipy_stats
                    t_stat, p_value = scipy_stats.ttest_ind(
                        male_valid, female_valid, equal_var=False
                    )
                    stats['t_statistic'] = float(t_stat)
                    stats['p_value_ttest'] = float(p_value)
                except:
                    stats['t_statistic'] = None
                    stats['p_value_ttest'] = None

        elif question_type == 'multiple_choice':
            # Count frequencies of each choice
            male_counter = Counter(male_valid)
            female_counter = Counter(female_valid)

            stats['male_distribution'] = dict(male_counter)
            stats['female_distribution'] = dict(female_counter)

            # Calculate most common response for each
            if male_counter:
                stats['male_mode'] = male_counter.most_common(1)[0][0]
                stats['male_mode_percent'] = (
                    male_counter.most_common(1)[0][1] / len(male_valid) * 100
                )

            if female_counter:
                stats['female_mode'] = female_counter.most_common(1)[0][0]
                stats['female_mode_percent'] = (
                    female_counter.most_common(1)[0][1] / len(female_valid) * 100
                )

            # Chi-square test for overall distribution
            if SCIPY_AVAILABLE and len(male_valid) >= 5 and len(female_valid) >= 5:
                try:
                    from scipy import stats as scipy_stats
                    # Build contingency table
                    all_choices = sorted(set(male_valid + female_valid))
                    male_counts = [male_counter.get(c, 0) for c in all_choices]
                    female_counts = [female_counter.get(c, 0) for c in all_choices]

                    contingency = [male_counts, female_counts]
                    chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
                    stats['chi2_statistic'] = float(chi2)
                    stats['p_value_chi2'] = float(p_value)
                    stats['chi2_dof'] = int(dof)
                except:
                    stats['chi2_statistic'] = None
                    stats['p_value_chi2'] = None

            # Fisher's exact test: strong treatment (C/D) vs weak treatment (A/B)
            # This tests the hypothesis that females receive weaker treatment
            if FISHER_EXACT_AVAILABLE:
                try:
                    male_strong = sum(male_counter.get(c, 0) for c in ['C', 'D'])
                    male_weak = sum(male_counter.get(c, 0) for c in ['A', 'B'])
                    female_strong = sum(female_counter.get(c, 0) for c in ['C', 'D'])
                    female_weak = sum(female_counter.get(c, 0) for c in ['A', 'B'])

                    contingency = [[male_strong, male_weak], [female_strong, female_weak]]

                    # One-sided test: alternative='less' tests if female has LESS strong treatment
                    odds_ratio, p_value = scipy_stats.fisher_exact(contingency, alternative='less')
                    stats['fisher_exact_odds_ratio'] = float(odds_ratio)
                    stats['p_value_fisher'] = float(p_value)
                    stats['fisher_contingency'] = {
                        'male_strong_CD': male_strong,
                        'male_weak_AB': male_weak,
                        'female_strong_CD': female_strong,
                        'female_weak_AB': female_weak
                    }
                except:
                    stats['fisher_exact_odds_ratio'] = None
                    stats['p_value_fisher'] = None

        return stats

    def fill_nulls(
        self,
        json_path: str,
        num_threads: int = 1
    ) -> Dict:
        """
        Fill in null responses in an existing results JSON file.

        Args:
            json_path: Path to the results JSON file
            num_threads: Number of parallel threads for API calls

        Returns:
            Updated results dictionary
        """
        # Load existing results
        with open(json_path, 'r') as f:
            results = json.load(f)

        experiment_key = results['experiment']
        model = results['model']
        question_id = results['question_id']
        temperature = results['temperature']
        max_tokens = results.get('max_tokens')

        model_short = model.split('/')[-1]

        # Find indices where responses are None
        male_null_indices = [i for i, r in enumerate(results['male_responses']) if r is None]
        female_null_indices = [i for i, r in enumerate(results['female_responses']) if r is None]

        # Also find indices where parsed is None but response exists (parse failures)
        for i, (resp, parsed) in enumerate(zip(results['male_responses'], results['male_parsed'])):
            if resp is not None and parsed is None and i not in male_null_indices:
                male_null_indices.append(i)
        for i, (resp, parsed) in enumerate(zip(results['female_responses'], results['female_parsed'])):
            if resp is not None and parsed is None and i not in female_null_indices:
                female_null_indices.append(i)

        total_nulls = len(male_null_indices) + len(female_null_indices)
        if total_nulls == 0:
            print(f"[{model_short}] No nulls to fill!")
            return results

        print(f"[{model_short}|{question_id}] Filling {len(male_null_indices)}M + {len(female_null_indices)}F = {total_nulls} nulls...")

        # Build trial list
        trial_args = []
        for idx in male_null_indices:
            trial_args.append(('male', idx))
        for idx in female_null_indices:
            trial_args.append(('female', idx))

        def run_fill_trial(args):
            """Worker function for filling a single null."""
            gender, idx = args
            response, metadata = self.run_single_trial(
                experiment_key, gender, model, question_id,
                temperature, max_tokens, None
            )
            return gender, idx, response, metadata

        # Execute trials
        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(run_fill_trial, args): args for args in trial_args}

                with tqdm(total=len(trial_args), desc=f"{model_short} fill", unit="t", ncols=60, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}') as pbar:
                    for future in as_completed(futures):
                        gender, idx, response, metadata = future.result()
                        # Update results in place
                        if gender == 'male':
                            results['male_responses'][idx] = response
                            results['male_metadata'][idx] = metadata
                        else:
                            results['female_responses'][idx] = response
                            results['female_metadata'][idx] = metadata
                        pbar.update(1)
        else:
            with tqdm(total=len(trial_args), desc=f"{model_short} fill", unit="t", ncols=60, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}') as pbar:
                for args in trial_args:
                    gender, idx, response, metadata = run_fill_trial(args)
                    if gender == 'male':
                        results['male_responses'][idx] = response
                        results['male_metadata'][idx] = metadata
                    else:
                        results['female_responses'][idx] = response
                        results['female_metadata'][idx] = metadata
                    pbar.update(1)

        # Re-parse all responses
        experiment = self.vignettes['experiments'][experiment_key]
        question = next(q for q in experiment['questions'] if q['id'] == question_id)

        results['male_unparsed_samples'] = []
        results['female_unparsed_samples'] = []
        results['male_parsed'] = self._parse_responses(
            results['male_responses'],
            question['type'],
            results['male_unparsed_samples']
        )
        results['female_parsed'] = self._parse_responses(
            results['female_responses'],
            question['type'],
            results['female_unparsed_samples']
        )

        # Recalculate statistics
        results['statistics'] = self._calculate_statistics(
            results['male_parsed'],
            results['female_parsed'],
            question['type']
        )

        # Save updated results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Report remaining nulls
        remaining_male = sum(1 for x in results['male_parsed'] if x is None)
        remaining_female = sum(1 for x in results['female_parsed'] if x is None)
        print(f"[{model_short}|{question_id}] Done. Remaining nulls: {remaining_male}M + {remaining_female}F")

        return results

    def save_results(self, results: Dict, output_path: str):
        """Save experiment results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    def print_summary(self, results: Dict):
        """Print a minimal summary of the experiment results."""
        stats = results['statistics']
        model_short = results['model'].split('/')[-1]

        # One-line summary
        m_valid = stats['male_valid_responses']
        f_valid = stats['female_valid_responses']
        total = results['num_trials']

        # Get question type
        experiment = self.vignettes['experiments'][results['experiment']]
        question = next(q for q in experiment['questions'] if q['id'] == results['question_id'])

        if question['type'] == 'rating':
            m_mean = stats.get('male_mean', 0) or 0
            f_mean = stats.get('female_mean', 0) or 0
            p_val = stats.get('p_value_ttest')
            p_str = f"p={p_val:.4f}" if p_val else "p=N/A"
            sig = "*" if p_val and p_val < 0.05 else ""
            print(f"[{model_short}|{results['question_id']}] Done: M={m_mean:.2f} F={f_mean:.2f} {p_str}{sig} ({m_valid}/{f_valid} valid)")
        else:
            m_mode = stats.get('male_mode', '?')
            f_mode = stats.get('female_mode', '?')
            p_val = stats.get('p_value_fisher') or stats.get('p_value_chi2')
            p_str = f"p={p_val:.4f}" if p_val else "p=N/A"
            sig = "*" if p_val and p_val < 0.05 else ""
            print(f"[{model_short}|{results['question_id']}] Done: M={m_mode} F={f_mode} {p_str}{sig} ({m_valid}/{f_valid} valid)")


def main():
    """Main entry point for the experiment harness."""
    parser = argparse.ArgumentParser(
        description='Run LLM medical diagnosis bias experiments'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=[
            'pain_management_kidney_stone',
            'pain_management_back_pain',
            'coronary_heart_disease'
        ],
        help='Which experiment to run (required unless --fill-nulls)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='OpenRouter model identifier (required unless --fill-nulls)'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Question ID to test (required unless --fill-nulls)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=30,
        help='Number of trials per gender (default: 30)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Maximum tokens to generate (default: None/unlimited)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: auto-generated)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenRouter API key (or set OPENROUTER_API_KEY env var)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between API calls in seconds (default: 1.0, only used if threads=1)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Number of parallel threads for API calls (default: 1)'
    )
    parser.add_argument(
        '--fill-nulls',
        type=str,
        metavar='JSON_FILE',
        help='Fill in null responses in an existing results JSON file (ignores other args except --threads and --api-key)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OpenRouter API key required. Set --api-key or OPENROUTER_API_KEY env var")
        return 1

    # Check for scipy
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not installed. Statistical tests will not be available.")
        print("Install with: pip install scipy")

    # Initialize harness
    harness = ExperimentHarness(api_key)

    # Handle fill-nulls mode
    if args.fill_nulls:
        harness.fill_nulls(args.fill_nulls, num_threads=args.threads)
        return 0

    # Validate required args for normal mode
    if not args.experiment or not args.model or not args.question:
        print("Error: --experiment, --model, and --question are required (unless using --fill-nulls)")
        return 1

    # Run experiment
    results = harness.run_experiment(
        experiment_key=args.experiment,
        model=args.model,
        question_id=args.question,
        num_trials=args.trials,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        delay_between_calls=args.delay,
        num_threads=args.threads
    )

    # Print summary
    harness.print_summary(results)

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_slug = args.model.replace('/', '_')
        output_path = f"results_{args.experiment}_{model_slug}_{args.question}_{timestamp}.json"

    harness.save_results(results, output_path)

    return 0


if __name__ == '__main__':
    exit(main())
