"""Benchmark runners for BBQ, CrowS-Pairs, and WinoBias (lightweight demo versions).

IMPORTANT (2026-09-19 webapp-parity remediation): these are NOT real BBQ /
CrowS-Pairs / WinoBias evaluations. There is no model, no dataset, and no
actual QA/stereotype/pronoun test being run here. Every "score" is a plain
arithmetic formula (base_score - bias_level * constant) driven entirely by
the bias_level slider the user sets. Previously this was displayed with the
real benchmark names and a PASS/FAIL verdict with zero indication that it was
synthetic -- indistinguishable from a genuine third-party benchmark run.
Every result now carries an explicit `is_simulated: True` flag and a
`methodology` note so it can never be mistaken for a real evaluation on
actual model outputs. This mirrors the F3 fix (remove/disclose fabricated
numbers) already applied to the MCP server on 2026-09-18.
"""
from typing import Dict, List
import numpy as np
from .utils import set_seed

_METHODOLOGY_NOTE = (
    "SIMULATED, not a real benchmark run: score = base_score - (bias_level * "
    "constant), a deterministic formula driven only by the bias_level slider. "
    "No model or dataset was actually evaluated. For training/demo purposes only."
)

def run_benchmark(benchmark: str, bias_level: float = 0.25, random_seed: int = 42) -> Dict:
    """
    Run a single SIMULATED benchmark (see module docstring: this does not
    execute a real BBQ/CrowS-Pairs/WinoBias evaluation).

    Returns structured result dict with score, threshold, pass_fail, details,
    is_simulated, methodology.
    """
    set_seed(random_seed)
    if benchmark == "BBQ":
        # Ambiguous context QA bias (names suggesting demographics)
        base_score = 0.78
        biased_score = base_score - (bias_level * 0.32)
        threshold = 0.65
        result = {
            "benchmark": "BBQ (simulated)",
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
        result = {
            "benchmark": "CrowS-Pairs (simulated)",
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
        result = {
            "benchmark": "WinoBias (simulated)",
            "score": round(biased_score, 3),
            "threshold": threshold,
            "pass_fail": "PASS" if biased_score >= threshold else "FAIL",
            "details": f"Pronoun resolution accuracy degraded under gender stereotype conditions"
        }
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    result["is_simulated"] = True
    result["methodology"] = _METHODOLOGY_NOTE
    return result

def run_all_benchmarks(bias_level: float = 0.25, random_seed: int = 42) -> List[Dict]:
    """Run all three SIMULATED benchmarks and return list of results."""
    results = []
    for b in ["BBQ", "CrowS-Pairs", "WinoBias"]:
        results.append(run_benchmark(b, bias_level, random_seed))
    return results