#!/usr/bin/env python3
"""
AI Bias Laboratory — Model Context Protocol (MCP) Server.
Enables Claude Desktop, Cursor, and OpenClaw to perform automated algorithmic fairness audits,
dataset bias scanning, Fairlearn mitigations, and EU AI Act Article 10 compliance evaluations.
"""

import sys
import os
import json
import uuid
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from ai_bias_lab.models import (
    FairnessMetrics, DatasetAuditInput, ModelFairnessInput,
    BiasAuditReport, LLMPromptAuditInput, ComplianceStatus
)
from ai_bias_lab.engine import FairnessAuditEngine
from ai_bias_lab.pension_scenarios import generate_pension_underwriting_dataset
from ai_bias_lab.llm_bias import LLMBiasAuditor
from ai_bias_lab.dashboard_generator import BiasDashboardGenerator

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "ai-bias-laboratory-mcp",
    "version": "1.0.0"
}

MAX_RECORDS_LIMIT = 50000

TOOLS = [
    {
        "name": "audit_dataset_bias",
        "description": (
            "Scans tabular dataset records for Disparate Impact, Statistical Parity Difference, and "
            "proxy-variable correlations against sensitive demographic attributes. Unknown target_column "
            "or sensitive_column values return a tool error listing available columns (no silent fallback). "
            "Verdicts report INSUFFICIENT_DATA when any compared subgroup has fewer than 30 records, and "
            "include subgroup_counts + bootstrap 95% confidence intervals. NOTE on proxy_correlations: only "
            "pairwise association between the sensitive attribute and each other single column is tested "
            "(Pearson for numeric-vs-numeric, Cramer's V / correlation ratio for categorical); feature "
            "INTERACTIONS that jointly reconstruct a protected attribute (e.g. two features that only "
            "together correlate with the sensitive attribute) will NOT be detected by this tool."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Name of dataset (e.g. 'Pension_Underwriting_2026')"},
                "target_column": {"type": "string", "description": "Binary outcome column name (1: approved/favorable, 0: rejected)"},
                "sensitive_column": {"type": "string", "description": "Sensitive attribute column (e.g. 'gender', 'postcode_cluster', 'age')"},
                "records": {
                    "type": "array",
                    "description": "Array of row dictionaries. If omitted, uses built-in Life & Pensions underwriting benchmark dataset.",
                    "items": {"type": "object"}
                },
                "favorable_outcome": {"description": "Value representing favorable outcome", "default": 1}
            },
            "required": ["dataset_name", "target_column", "sensitive_column"]
        }
    },
    {
        "name": "evaluate_model_fairness",
        "description": "Evaluates model predictions (y_pred) against ground truth labels (y_true) and sensitive features for Equalized Odds and Disparate Impact.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "y_true": {"type": "array", "items": {"type": "integer"}},
                "y_pred": {"type": "array", "items": {"type": "integer"}},
                "sensitive_features": {"type": "array"},
                "sensitive_feature_name": {"type": "string", "default": "gender"}
            },
            "required": ["model_name", "y_true", "y_pred", "sensitive_features"]
        }
    },
    {
        "name": "suggest_fairness_mitigation",
        "description": "Recommends actionable pre-processing (reweighting), in-processing, and post-processing (ThresholdOptimizer) mitigations to satisfy EU AI Act Art. 10 and the EEOC 80% rule.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "disparate_impact_ratio": {"type": "number", "description": "Current Disparate Impact Ratio"},
                "statistical_parity_difference": {"type": "number", "description": "Current Statistical Parity Difference"},
                "equalized_odds_difference": {"type": "number", "description": "Current Equalized Odds Difference"}
            },
            "required": ["disparate_impact_ratio", "statistical_parity_difference", "equalized_odds_difference"]
        }
    },
    {
        "name": "audit_llm_prompt_bias",
        "description": "Tests a Generative AI / LLM system prompt across demographic customer personas in Life & Pensions for sentiment variance and decision disparity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "system_prompt": {"type": "string", "description": "System prompt text for claims/underwriting LLM"},
                "scenario_name": {"type": "string", "default": "Pension Claims Triage"}
            },
            "required": ["system_prompt"]
        }
    },
    {
        "name": "generate_bias_audit_report",
        "description": "Generates a formal standalone HTML audit report and certificate for EU AI Act Article 10 compliance dossiers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "system_name": {"type": "string"},
                "domain": {"type": "string", "default": "Life & Pensions Underwriting"},
                "output_path": {"type": "string", "description": "Optional output path for HTML report"}
            },
            "required": ["system_name"]
        }
    }
]

_ENGINE = None

def get_engine() -> FairnessAuditEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FairnessAuditEngine()
    return _ENGINE


def handle_request(req):
    # Guard against non-dict requests
    if not isinstance(req, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request: root payload must be a JSON object"}
        }

    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if not isinstance(method, str):
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32600, "message": "Invalid Request: 'method' must be a string"}
        }

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO
            }
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS}
        }
    elif method == "tools/call":
        if not isinstance(params, dict):
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: expected object"}}
        tool_name = params.get("name")
        if not tool_name or not isinstance(tool_name, str):
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'name' is required"}}

        args = params.get("arguments", {}) or {}
        if not isinstance(args, dict):
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'arguments' must be an object"}}

        engine = get_engine()

        try:
            if tool_name == "audit_dataset_bias":
                dataset_name = args.get("dataset_name")
                target_col = args.get("target_column")
                sens_col = args.get("sensitive_column")
                records = args.get("records")

                # FIX F1 (2026-09-18 remediation brief, P0): the tool used to
                # silently substitute a default column ('y_pred_approval' /
                # 'gender') whenever the caller's requested column name did
                # not exist on the BUILT-IN dataset, then report results
                # under a column name the caller never asked for -- with no
                # error, no warning field. A caller who mistyped a column
                # name (e.g. 'age_band') would get back plausible,
                # correctly-formatted, WRONGLY LABELLED numbers (silently
                # analysing 'gender' while believing they analysed something
                # else). There must be no code path where the analysed
                # column differs from the requested column. Every default
                # fallback path has been removed.
                if not isinstance(target_col, str) or not target_col.strip():
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'target_column' is required and must be a non-empty string"}}
                if not isinstance(sens_col, str) or not sens_col.strip():
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'sensitive_column' is required and must be a non-empty string"}}

                if records is not None and not isinstance(records, list):
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'records' must be an array"}}
                if records and len(records) > MAX_RECORDS_LIMIT:
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Record limit exceeded (max {MAX_RECORDS_LIMIT})"}}

                if records:
                    df = pd.DataFrame(records)
                else:
                    df = generate_pension_underwriting_dataset(n_samples=500)

                # Validate BOTH requested columns against the actual dataset
                # columns before any computation happens. Report every
                # unknown column by name (not just the first one found) plus
                # the full list of columns that were actually available, so
                # a typo is caught immediately instead of silently
                # mislabelling a different attribute.
                available_columns = list(df.columns)
                unknown_cols = [c for c in (target_col, sens_col) if c not in available_columns]
                if unknown_cols:
                    if len(unknown_cols) == 1:
                        col_desc = f"Unknown column '{unknown_cols[0]}'"
                    else:
                        col_desc = "Unknown columns: " + ", ".join(f"'{c}'" for c in unknown_cols)
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"{col_desc}. Available columns: [{', '.join(available_columns)}]"}}

                y_pred = df[target_col].values
                sens = df[sens_col].values
                y_true = df["y_true_eligibility"].values if "y_true_eligibility" in df.columns else y_pred

                metrics = engine.calculate_fairness_metrics(y_true, y_pred, sens)
                proxy_corrs = engine.detect_proxy_correlations(df, sens_col)
                mitigations = engine.suggest_mitigations(metrics)

                report = {
                    "dataset_name": dataset_name,
                    "records_analyzed": len(df),
                    "sensitive_attribute": sens_col,
                    "metrics": metrics.model_dump(),
                    "proxy_correlations": proxy_corrs.model_dump(),
                    "mitigation_recommendations": [m.model_dump() for m in mitigations],
                    "eu_ai_act_art10_verdict": metrics.eu_ai_act_art10_status.value
                }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(report, indent=2)}], "isError": False}
                }

            elif tool_name == "evaluate_model_fairness":
                model_name = args.get("model_name")
                y_true = args.get("y_true")
                y_pred = args.get("y_pred")
                sens = args.get("sensitive_features")

                if not isinstance(y_true, list) or not isinstance(y_pred, list) or not isinstance(sens, list):
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: arrays required"}}
                if len(y_true) != len(y_pred) or len(y_true) != len(sens):
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Array lengths must match"}}

                metrics = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
                mitigations = engine.suggest_mitigations(metrics)

                res = {
                    "model_name": model_name,
                    "sample_count": len(y_true),
                    "metrics": metrics.model_dump(),
                    "mitigations": [m.model_dump() for m in mitigations]
                }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}], "isError": False}
                }

            elif tool_name == "suggest_fairness_mitigation":
                di = float(args.get("disparate_impact_ratio", 1.0))
                spd = float(args.get("statistical_parity_difference", 0.0))
                eod = float(args.get("equalized_odds_difference", 0.0))

                dummy_metrics = FairnessMetrics(
                    statistical_parity_difference=spd,
                    equalized_odds_difference=eod,
                    disparate_impact_ratio=di,
                    bias_amplification_factor=1.0,
                    disparate_impact_status=ComplianceStatus.COMPLIANT if di >= 0.80 else ComplianceStatus.NON_COMPLIANT,
                    # FIX F5: renamed from eu_ai_act_art10_status (mislabelled;
                    # this is the US EEOC four-fifths threshold, not an EU AI
                    # Act Article 10 verdict). Deprecated alias auto-synced.
                    eeoc_four_fifths_status=ComplianceStatus.COMPLIANT if di >= 0.80 and abs(spd) <= 0.10 else ComplianceStatus.NON_COMPLIANT
                )
                recs = engine.suggest_mitigations(dummy_metrics)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps([r.model_dump() for r in recs], indent=2)}], "isError": False}
                }

            elif tool_name == "audit_llm_prompt_bias":
                system_prompt = args.get("system_prompt")
                if not system_prompt or not isinstance(system_prompt, str):
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'system_prompt' is required"}}
                scenario = args.get("scenario_name", "Pension Claims Triage")

                auditor = LLMBiasAuditor()
                report = auditor.audit_prompt_personas(system_prompt, scenario_name=scenario)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(report.model_dump(), indent=2)}], "isError": False}
                }

            elif tool_name == "generate_bias_audit_report":
                system_name = args.get("system_name")
                domain = args.get("domain", "Life & Pensions Underwriting")
                out_path = args.get("output_path")

                # Run baseline pension benchmark
                df = generate_pension_underwriting_dataset(n_samples=500)
                metrics = engine.calculate_fairness_metrics(df["y_true_eligibility"].values, df["y_pred_approval"].values, df["gender"].values)
                proxy_corrs = engine.detect_proxy_correlations(df, "gender")
                mitigations = engine.suggest_mitigations(metrics)

                audit_rep = BiasAuditReport(
                    audit_id=f"BIAS-AUD-{uuid.uuid4().hex[:6].upper()}",
                    system_name=system_name,
                    domain=domain,
                    evaluated_at="2026-09-18",
                    metrics=metrics,
                    detected_proxy_correlations=proxy_corrs,
                    # FIX F5: renamed key from EU_AI_Act_Article_10 to
                    # EEOC_Four_Fifths_Rule -- this verdict is the US EEOC
                    # four-fifths threshold, not an EU AI Act Article 10
                    # verdict. Art. 10's actual documentation checklist is
                    # carried separately on metrics.art10_documentation_status
                    # (not a regulatory_verdicts PASS/FAIL, since it is a
                    # checklist of not_assessed/assessed items, not a ratio).
                    regulatory_verdicts={
                        "EEOC_Four_Fifths_Rule": metrics.eeoc_four_fifths_status.value,
                        "WGBU_Equal_Treatment": "PASS" if metrics.disparate_impact_ratio >= 0.80 else "FAIL",
                        "EEOC_80_Percent_Rule": "PASS" if metrics.disparate_impact_ratio >= 0.80 else "FAIL"
                    },
                    mitigation_recommendations=mitigations,
                    executive_summary=(
                        f"Algorithmic fairness assessment of {system_name} against the US EEOC four-fifths rule. "
                        f"Disparate impact ratio is {metrics.disparate_impact_ratio} with statistical parity difference of {metrics.statistical_parity_difference}. "
                        f"EU AI Act Article 10 data-governance documentation status: not independently assessed by this tool (see art10_documentation_status)."
                    )
                )

                html_out = BiasDashboardGenerator.generate_html(audit_rep, out_path)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Bias report successfully generated for {system_name}. EEOC four-fifths verdict: {metrics.eeoc_four_fifths_status.value}."}],
                        "html_preview": html_out[:400] + "...",
                        "isError": False
                    }
                }

            else:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}

        except Exception as e:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": f"Internal error: {str(e)}"}}

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def main():
    """Main stdio loop for MCP Server."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            if res is not None:
                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error: Invalid JSON"}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
        except Exception as ex:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": "Internal process error"}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
