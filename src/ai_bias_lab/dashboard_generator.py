"""
HTML & Markdown Report Generator for AI Bias Laboratory.
Generates EU AI Act Article 10 & ISO 42001 Algorithmic Fairness Audit Reports.
"""

import html
import os
from typing import Optional
from ai_bias_lab.models import BiasAuditReport, ComplianceStatus


class BiasDashboardGenerator:
    """
    Generates interactive standalone HTML audit reports with SVG charts,
    disparate impact indicators, and regulatory compliance verdicts.
    """

    @staticmethod
    def generate_html(report: BiasAuditReport, out_path: Optional[str] = None) -> str:
        safe_system_name = html.escape(str(report.system_name))
        safe_domain = html.escape(str(report.domain))
        safe_summary = html.escape(str(report.executive_summary))

        m = report.metrics
        # FIX N7 / FIX N4 (round 2 verification, 2026-09-22): both fields can
        # now be None when not computable (undefined equalized-odds rate,
        # or a ground-truth disparity too close to zero for
        # bias_amplification_factor -- see warnings). Render that
        # explicitly rather than the bare Python string "None".
        eod_display = m.equalized_odds_difference if m.equalized_odds_difference is not None else "Not computable (see warnings)"
        di_color = "#16A34A" if m.disparate_impact_status == ComplianceStatus.COMPLIANT else ("#D97706" if m.disparate_impact_status == ComplianceStatus.WARNING else "#DC2626")
        # FIX F5: renamed from eu_ai_act_art10_status (mislabelled -- this is
        # the US EEOC four-fifths threshold, not an EU AI Act Article 10 verdict).
        eeoc_color = "#16A34A" if m.eeoc_four_fifths_status == ComplianceStatus.COMPLIANT else ("#D97706" if m.eeoc_four_fifths_status == ComplianceStatus.WARNING else "#DC2626")

        # Build proxy correlation rows
        # FIX F4: report.detected_proxy_correlations is now a
        # ProxyCorrelationResult, not a bare dict. Render its
        # computed/not_computed status explicitly instead of letting an
        # empty correlations dict look identical to 'nothing was checked'.
        proxy_result = report.detected_proxy_correlations
        proxy_rows = []
        if proxy_result.status == "not_computed":
            proxy_rows.append(
                f"<tr><td colspan='3' style='text-align:center; color:#92400E; background:#FEF3C7;'>"
                f"Proxy-detectie niet uitgevoerd: {html.escape(str(proxy_result.reason or 'onbekende reden'))}</td></tr>"
            )
        else:
            for feat, corr in proxy_result.correlations.items():
                proxy_rows.append(f"""
                <tr>
                    <td><strong>{html.escape(str(feat))}</strong></td>
                    <td><code>{corr}</code></td>
                    <td><span class="badge" style="background:#FEF3C7; color:#92400E;">High Proxy Risk (>0.35)</span></td>
                </tr>
                """)
            if not proxy_rows:
                proxy_rows.append(f"<tr><td colspan='3' style='text-align:center; color:#6B7280;'>Geen significante proxy-correlaties gedetecteerd (drempelwaarde < 0.35; methode: {html.escape(str(proxy_result.method or 'n.v.t.'))}).</td></tr>")

        # Build mitigation rows
        mit_rows = []
        for rec in report.mitigation_recommendations:
            mit_rows.append(f"""
            <tr>
                <td><strong>{html.escape(str(rec.mitigation_type.value))}</strong></td>
                <td><code>{rec.current_value}</code> &rarr; <code>{rec.target_value}</code></td>
                <td>{html.escape(str(rec.recommended_action))}</td>
                <td><span class="badge" style="background:#DCFCE7; color:#166534;">gap: {rec.heuristic_target_gap}</span></td>
            </tr>
            """)
        if not mit_rows:
            mit_rows.append("<tr><td colspan='4' style='text-align:center; color:#16A34A;'>Model voldoet aan alle gestelde fairness normen; geen directe mitigatie vereist.</td></tr>")

        html_content = f"""<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Fairness & Bias Audit Report — {safe_system_name}</title>
    <style>
        :root {{
            --primary: #1A365D;
            --primary-light: #2B6CB0;
            --bg: #F8FAFC;
            --card-bg: #FFFFFF;
            --text: #1E293B;
            --border: #E2E8F0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%);
            color: white;
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card-title {{ font-size: 12px; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 8px; }}
        .card-value {{ font-size: 30px; font-weight: 700; color: var(--primary); }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }}
        table {{ width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 10px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 24px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); font-size: 14px; }}
        th {{ background-color: #F1F5F9; font-size: 12px; font-weight: 700; text-transform: uppercase; color: #475569; }}
        tr:last-child td {{ border-bottom: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚖️ Algorithmic Fairness & Bias Audit Report</h1>
            <p>System: <strong>{safe_system_name}</strong> | Domain: <strong>{safe_domain}</strong> | Audit ID: <code>{html.escape(report.audit_id)}</code></p>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">Disparate Impact Ratio (80% Rule)</div>
                <div class="card-value" style="color:{di_color};">{m.disparate_impact_ratio}</div>
                <div style="font-size:12px; margin-top:4px;">Status: <span class="badge" style="background:{di_color}; color:#fff;">{m.disparate_impact_status.value}</span></div>
            </div>
            <div class="card">
                <div class="card-title">EEOC Four-Fifths Rule ({html.escape(m.threshold_source)})</div>
                <div class="card-value" style="color:{eeoc_color};">{m.eeoc_four_fifths_status.value}</div>
                <div style="font-size:12px; margin-top:4px;">Data Governance & Non-discrimination</div>
            </div>
            <div class="card">
                <div class="card-title">Statistical Parity Difference</div>
                <div class="card-value">{m.statistical_parity_difference}</div>
                <div style="font-size:12px; color:#6B7280; margin-top:4px;">Selection rate gap</div>
            </div>
            <div class="card">
                <div class="card-title">Equalized Odds Difference</div>
                <div class="card-value">{eod_display}</div>
                <div style="font-size:12px; color:#6B7280; margin-top:4px;">Error rate parity (TPR/FPR)</div>
            </div>
        </div>

        <div class="card" style="margin-bottom:24px;">
            <div class="card-title">Executive Summary & Audit Context</div>
            <p style="margin:0; line-height:1.6;">{safe_summary}</p>
        </div>

        <div class="card" style="padding:0; overflow:hidden; margin-bottom:24px;">
            <div style="padding:16px; font-weight:700; border-bottom:1px solid var(--border); background:#F8FAFC;">🔍 Gedetecteerde Proxy-Discriminatie Correlaties</div>
            <table>
                <thead>
                    <tr><th>Feature / Variabele</th><th>Correlatie met Beschermd Kenmerk</th><th>Risico Classificatie</th></tr>
                </thead>
                <tbody>
                    {''.join(proxy_rows)}
                </tbody>
            </table>
        </div>

        <div class="card" style="padding:0; overflow:hidden;">
            <div style="padding:16px; font-weight:700; border-bottom:1px solid var(--border); background:#F8FAFC;">🛠️ Aanbevolen Fairlearn Mitigatiemaatregelen</div>
            <table>
                <thead>
                    <tr><th>Type Mitigatie</th><th>Huidig &rarr; Doel</th><th>Aanbevolen Actie</th><th>Verwachte Winst</th></tr>
                </thead>
                <tbody>
                    {''.join(mit_rows)}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        if out_path:
            clean_path = os.path.abspath(out_path)
            with open(clean_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content
