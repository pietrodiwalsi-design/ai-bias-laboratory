"""
Phase 3 Tests for AI Bias Laboratory — FastMCP Protocol & Hardening.
"""

import pytest
import json
from ai_bias_lab.mcp_server import handle_request, PROTOCOL_VERSION, SERVER_INFO, TOOLS


def test_mcp_initialize():
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {}
        }
    }
    res = handle_request(req)
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 1
    assert res["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert res["result"]["serverInfo"]["name"] == "ai-bias-laboratory-mcp"


def test_mcp_tools_list():
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    res = handle_request(req)
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 2
    tools = res["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "audit_dataset_bias" in tool_names
    assert "evaluate_model_fairness" in tool_names
    assert "suggest_fairness_mitigation" in tool_names
    assert "audit_llm_prompt_bias" in tool_names
    assert "generate_bias_audit_report" in tool_names


def test_mcp_audit_dataset_bias_call():
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "audit_dataset_bias",
            "arguments": {
                "dataset_name": "Pension_Underwriting_2026",
                "target_column": "y_pred_approval",
                "sensitive_column": "gender"
            }
        }
    }
    res = handle_request(req)
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 3
    assert not res["result"]["isError"]
    content = json.loads(res["result"]["content"][0]["text"])
    assert content["dataset_name"] == "Pension_Underwriting_2026"
    assert "disparate_impact_ratio" in content["metrics"]
    assert "eu_ai_act_art10_verdict" in content


def test_mcp_non_dict_payload_guard():
    res1 = handle_request("plain_string")
    assert res1["error"]["code"] == -32600
    res2 = handle_request([1, 2, 3])
    assert res2["error"]["code"] == -32600


def test_mcp_invalid_tool_error():
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "non_existent_tool",
            "arguments": {}
        }
    }
    res = handle_request(req)
    assert "error" in res
    assert res["error"]["code"] == -32601
