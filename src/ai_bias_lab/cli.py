"""
CLI interface for AI Bias Laboratory.
"""

import sys
import argparse
import json
import pandas as pd
import numpy as np

from ai_bias_lab.engine import FairnessAuditEngine
from ai_bias_lab.pension_scenarios import generate_pension_underwriting_dataset
from ai_bias_lab.llm_bias import LLMBiasAuditor
from ai_bias_lab.dashboard_generator import BiasDashboardGenerator
from ai_bias_lab.models import BiasAuditReport


def main():
    parser = argparse.ArgumentParser(description="AI Bias Laboratory CLI")
    parser.add_argument("--demo", action="store_true", help="Run Life & Pensions Underwriting bias audit demo")
    parser.add_argument("--output-html", type=str, default="ai_bias_audit_report.html", help="HTML report output path")
    parser.add_argument("--llm-prompt", type=str, help="Audit an LLM system prompt against pension personas")
    parser.add_argument("--json", action="store_true", help="Output JSON results")

    args = parser.parse_args()
    engine = FairnessAuditEngine()

    if args.llm_prompt:
        auditor = LLMBiasAuditor()
        report = auditor.audit_prompt_personas(args.llm_prompt)
        print(json.dumps(report.model_dump(), indent=2))
        return

    if args.demo:
        df = generate_pension_underwriting_dataset(n_samples=1000)
        y_true = df["y_true_eligibility"].values
        y_pred = df["y_pred_approval"].values
        sens = df["gender"].values

        metrics = engine.calculate_fairness_metrics(y_true, y_pred, sens)
        proxy_corrs = engine.detect_proxy_correlations(df, "gender")
        mitigations = engine.suggest_mitigations(metrics)

        audit_rep = BiasAuditReport(
            audit_id="BIAS-DEMO-2026",
            system_name="Life & Pensions Algorithmic Underwriting Engine",
            domain="Life & Pensions Insurance",
            evaluated_at="2026-09-18",
            metrics=metrics,
            detected_proxy_correlations=proxy_corrs,
            regulatory_verdicts={
                "EEOC_Four_Fifths_Rule": metrics.eeoc_four_fifths_status.value,
                "WGBU_Equal_Treatment": "PASS" if metrics.disparate_impact_ratio >= 0.80 else "FAIL",
                "EEOC_80_Percent_Rule": "PASS" if metrics.disparate_impact_ratio >= 0.80 else "FAIL"
            },
            mitigation_recommendations=mitigations,
            executive_summary=(
                "Algorithmic fairness assessment of the Life & Pensions underwriting engine. "
                f"Disparate impact ratio is {metrics.disparate_impact_ratio} with statistical parity difference of {metrics.statistical_parity_difference}."
            )
        )

        if args.json:
            print(json.dumps(audit_rep.model_dump(), indent=2))
        else:
            BiasDashboardGenerator.generate_html(audit_rep, args.output_html)
            print("✅ AI Bias Laboratory Audit Complete!")
            print(f"   Disparate Impact Ratio: {metrics.disparate_impact_ratio} ({metrics.disparate_impact_status.value})")
            print(f"   EEOC Four-Fifths Rule Verdict: {metrics.eeoc_four_fifths_status.value}")
            print(f"   EU AI Act Art. 10 documentation status: not independently assessed (see art10_documentation_status)")
            print(f"   HTML Report: {args.output_html}")


if __name__ == "__main__":
    main()
