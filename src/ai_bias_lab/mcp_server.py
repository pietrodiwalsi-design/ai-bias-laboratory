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
    BiasAuditReport, LLMPromptAuditInput, ComplianceStatus,
    ProxyCorrelationResult
)
from ai_bias_lab.engine import FairnessAuditEngine, compute_dataset_fingerprint, TOOL_VERSION
from datetime import datetime, timezone
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
            "together correlate with the sensitive attribute) will NOT be detected by this tool. NOTE on "
            "bias_amplification_factor: formula is abs(max_selection_rate - min_selection_rate) across "
            "predicted outcomes, divided by the SAME quantity across GROUND-TRUTH labels (floor 0.01 on the "
            "denominator to avoid division by zero). Range is theoretically unbounded (0 = model perfectly "
            "equalizes an unequal ground truth; 1.0 = model reproduces ground-truth disparity unchanged; "
            ">1.0 = model WIDENS a disparity that was smaller or absent in the ground truth; values near 0 "
            "with LOW disparate_impact_ratio, as with 'age', suggest the disparity pre-exists in the "
            "training data rather than being introduced by the model). This is a descriptive ratio, not a "
            "pass/fail threshold under any named regulation or standard -- no compliance verdict is derived "
            "from it."
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
                "favorable_outcome": {"description": "Value representing favorable outcome", "default": 1},
                "reference_group": {"type": "string", "description": "Optional: force this subgroup value as the reference/baseline for disparate_impact_ratio and statistical_parity_difference (both pairwise metrics). Defaults to the subgroup with the highest selection rate if omitted."}
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
        "description": (
            "Recommends actionable pre-processing (reweighting), in-processing, and post-processing "
            "(ThresholdOptimizer) mitigations to satisfy the EEOC 80% four-fifths rule. NOTE (F10): this is "
            "deterministic templating on threshold comparisons against the three numbers you pass in, not an "
            "independent analysis. audit_dataset_bias ALREADY embeds this exact same output as its "
            "mitigation_recommendations field for every audit it runs -- call THIS tool separately only when "
            "you already have metrics from elsewhere (a different tool, a manual calculation) and did not "
            "run audit_dataset_bias. If you just ran audit_dataset_bias, do not call this tool again for the "
            "same numbers; read mitigation_recommendations from that result instead."
        ),
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
        "description": (
            "Renders a standalone HTML report from an audit_result you already obtained from "
            "audit_dataset_bias -- it does NOT run a new audit or recompute anything. This is NOT a "
            "certificate or attestation of any kind; it is a formatted rendering of findings you supply. "
            "FIX F9 (2026-09-18 remediation brief): previously this tool ignored any prior audit and "
            "silently recomputed its own fixed baseline (built-in dataset, hardcoded sensitive_column='gender'), "
            "so a report could describe a completely different analysis than the one the caller actually "
            "reviewed. audit_result is now a mandatory parameter -- the exact JSON object returned by "
            "audit_dataset_bias -- and is rendered verbatim, including its dataset_fingerprint and "
            "audit_timestamp, so the report is traceable to the specific audit run it documents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "system_name": {"type": "string"},
                "domain": {"type": "string", "default": "Life & Pensions Underwriting"},
                "audit_result": {
                    "type": "object",
                    "description": "The full JSON object previously returned by audit_dataset_bias for this system. Required -- this tool renders that result, it does not recompute it."
                },
                "output_path": {"type": "string", "description": "Required output path for the HTML report. There is no default location -- the tool refuses to write anywhere the caller did not explicitly specify."}
            },
            "required": ["system_name", "audit_result", "output_path"]
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

                # FIX F6: optional caller-specified reference_group.
                reference_group = args.get("reference_group")
                if reference_group is not None and not isinstance(reference_group, str):
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'reference_group' must be a string"}}

                # FIX F8: optional caller-specified random_seed, echoed back
                # for bootstrap reproducibility. Defaults to the engine's
                # own seed (42) when omitted, same as before.
                random_seed_arg = args.get("random_seed")
                if random_seed_arg is not None and not isinstance(random_seed_arg, int):
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'random_seed' must be an integer"}}

                try:
                    metrics = engine.calculate_fairness_metrics(
                        y_true, y_pred, sens,
                        reference_group=reference_group,
                        random_seed=random_seed_arg,
                    )
                except ValueError as ve:
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": str(ve)}}
                proxy_corrs = engine.detect_proxy_correlations(df, sens_col)
                mitigations = engine.suggest_mitigations(metrics)

                # FIX F8: audit trail, so this output can be tied to a
                # specific data version and reproduced.
                metrics.audit_timestamp = datetime.now(timezone.utc).isoformat()
                metrics.tool_version = TOOL_VERSION
                metrics.dataset_fingerprint = compute_dataset_fingerprint(df)
                metrics.random_seed_used = random_seed_arg if random_seed_arg is not None else engine.random_seed

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
                # FIX F9 (2026-09-18 remediation brief, P2): this tool used
                # to accept NO audit findings as input -- signature was
                # (system_name, domain, output_path) -- and silently
                # recomputed its own fixed baseline internally (built-in
                # dataset, hardcoded sensitive_column='gender'), regardless
                # of what the caller had actually audited or reviewed. A
                # report for a postcode_cluster audit would render gender
                # figures instead, with no indication anything diverged.
                # audit_result is now REQUIRED and rendered verbatim -- this
                # tool no longer computes anything, it only formats an
                # already-obtained audit_dataset_bias result. output_path is
                # also now required: no default write location.
                system_name = args.get("system_name")
                domain = args.get("domain", "Life & Pensions Underwriting")
                out_path = args.get("output_path")
                audit_result = args.get("audit_result")

                if not isinstance(system_name, str) or not system_name.strip():
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'system_name' is required and must be a non-empty string"}}
                if not isinstance(out_path, str) or not out_path.strip():
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'output_path' is required. This tool refuses to pick a default write location."}}
                if not isinstance(audit_result, dict):
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'audit_result' is required and must be the JSON object previously returned by audit_dataset_bias. This tool renders a prior audit, it does not compute a new one."}}

                try:
                    metrics_dict = dict(audit_result["metrics"])
                    proxy_corrs_dict = dict(audit_result["proxy_correlations"])
                except (KeyError, TypeError) as e:
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Invalid params: 'audit_result' is missing expected field: {e}. It must be the unmodified object returned by audit_dataset_bias."}}

                try:
                    metrics = FairnessMetrics(**metrics_dict)
                    proxy_corrs = ProxyCorrelationResult(**proxy_corrs_dict)
                except Exception as e:
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Invalid params: 'audit_result' failed validation: {e}"}}

                mitigations = engine.suggest_mitigations(metrics)

                audit_rep = BiasAuditReport(
                    audit_id=f"BIAS-AUD-{uuid.uuid4().hex[:6].upper()}",
                    system_name=system_name,
                    domain=domain,
                    evaluated_at=metrics.audit_timestamp or "unknown",
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
                        f"Algorithmic fairness assessment of {system_name} against the US EEOC four-fifths rule, "
                        f"rendered from audit dataset_fingerprint={metrics.dataset_fingerprint}, audited at {metrics.audit_timestamp}. "
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
