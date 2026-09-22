"""
Life & Pensions Domain Scenarios & Synthetic Datasets for Fairness Auditing.
Covers Underwriting, Disability Claims Triage, and Pension Mortality / Fraud Detection.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np


def generate_pension_underwriting_dataset(
    n_samples: int = 1000,
    inject_proxy_bias: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic Life & Pensions individual underwriting dataset
    with demographic features, financial risk metrics, and optional proxy bias.
    """
    rng = np.random.default_rng(seed)

    # Demographic attributes
    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48])
    age = rng.integers(22, 67, size=n_samples)
    
    # Postcode clusters (proxy for socio-economic / ethnic demographics)
    postcode_risk_cluster = rng.choice(["A_Urban", "B_Suburban", "C_HighRiskUrban", "D_Rural"], size=n_samples, p=[0.35, 0.35, 0.20, 0.10])
    
    # Financial & employment features
    annual_income = rng.lognormal(mean=10.8, sigma=0.4, size=n_samples) # ~€45k - €80k
    pension_accrual_years = np.clip((age - 22) * rng.uniform(0.7, 1.0, size=n_samples), 0, 45).astype(int)
    pre_existing_conditions = rng.binomial(1, p=np.clip(0.05 + (age - 22) * 0.005, 0.05, 0.4), size=n_samples)
    
    # Ground truth: true actuarial risk score (0: standard, 1: high actuarial risk)
    true_actuarial_risk = (
        (age > 55).astype(int) * 0.3 +
        pre_existing_conditions * 0.5 +
        (annual_income < 30000).astype(int) * 0.2
    )
    y_true = (true_actuarial_risk + rng.normal(0, 0.1, n_samples) < 0.45).astype(int) # 1 = approved, 0 = rejected

    # Algorithmic decision (y_pred): potentially biased against Female or Postcode C
    model_score = 0.5 * (annual_income / 60000) + 0.3 * (pension_accrual_years / 20) - 0.4 * pre_existing_conditions
    
    if inject_proxy_bias:
        # Penalize Female (part-time accrual penalty proxy) and Postcode C
        gender_penalty = np.where(gender == "Female", -0.22, 0.0)
        postcode_penalty = np.where(postcode_risk_cluster == "C_HighRiskUrban", -0.30, 0.0)
        model_score += gender_penalty + postcode_penalty

    y_pred = (model_score + rng.normal(0, 0.15, n_samples) > 0.48).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"POL-{i+10000}" for i in range(n_samples)],
        "gender": gender,
        "age": age,
        "postcode_cluster": postcode_risk_cluster,
        "annual_income": np.round(annual_income, 2),
        "pension_accrual_years": pension_accrual_years,
        "pre_existing_conditions": pre_existing_conditions,
        "y_true_eligibility": y_true,
        "y_pred_approval": y_pred
    })

    return df
