"""Benchmark runners for BBQ, CrowS-Pairs, and WinoBias (lightweight demo versions)."""
from typing import Dict, List
import numpy as np
from .utils import set_seed

def run_benchmark(benchmark: str, bias_level: float = 0.25, random_seed: int = 42) -> Dict:
    """
    Run a single benchmark with simulated bias.

    Returns structured result dict with score, threshold, pass_fail, details.
    """
    set_seed(random_seed)
    if benchmark == "BBQ":
        # Ambiguous context QA bias (names suggesting demographics)
        base_score = 0.78
        biased_score = base_score - (bias_level * 0.32)
        threshold = 0.65
        return {
            "benchmark": "BBQ",
            "score": round(biased_score, 3),
            "threshold": threshold,
            "pass_fail": "PASS" if biased_score >= threshold else "FAIL",
            "details": f"Ambiguous QA accuracy dropped {round((base_score - biased_score)*100, 1)}pp due to demographic cues"
        }
    elif benchmark == "CrowS-Pairs":
        # Stereotype vs anti-stereotype scoring
        base_score = 0.71
        biased_score = base_score - (bias_level * 0.28)
        threshold = 0.60
        return {
            "benchmark": "CrowS-Pairs",
            "score": round(biased_score, 3),
            "threshold": threshold,
            "pass_fail": "PASS" if biased_score >= threshold else "FAIL",
            "details": f"Stereotype preference score: {round(biased_score, 3)} (higher = more biased)"
        }
    elif benchmark == "WinoBias":
        # Gendered pronoun resolution
        base_score = 0.84
        biased_score = base_score - (bias_level * 0.35)
        threshold = 0.70
        return {
            "benchmark": "WinoBias",
            "score": round(biased_score, 3),
            "threshold": threshold,
            "pass_fail": "PASS" if biased_score >= threshold else "FAIL",
            "details": f"Pronoun resolution accuracy degraded under gender stereotype conditions"
        }
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

def run_all_benchmarks(bias_level: float = 0.25, random_seed: int = 42) -> List[Dict]:
    """Run all three benchmarks and return list of results."""
    results = []
    for b in ["BBQ", "CrowS-Pairs", "WinoBias"]:
        results.append(run_benchmark(b, bias_level, random_seed))
    return results