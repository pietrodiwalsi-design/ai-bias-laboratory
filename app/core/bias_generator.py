"""Synthetic Bias Injection Engine for AI Bias Laboratory.

Generates controlled bias in datasets for testing fairness metrics.
"""
from typing import Dict, Tuple, Literal
import pandas as pd
import numpy as np
from .utils import set_seed, MIN_SUBGROUP_SAMPLE_SIZE

InjectionMethod = Literal["label_flip", "feature_skew", "sampling_bias"]

def inject_bias(
    model_data: pd.DataFrame,
    protected_attribute: str,
    bias_intensity: float = 0.3,
    injection_method: InjectionMethod = "label_flip",
    random_seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Inject synthetic bias into a dataset.

    Args:
        model_data: Input DataFrame with protected_attribute and outcome columns
        protected_attribute: Column name (gender/race/age/disability)
        bias_intensity: Strength of bias from 0.0 (none) to 1.0 (maximum)
        injection_method: One of label_flip, feature_skew, sampling_bias
        random_seed: For reproducibility

    Returns:
        (biased_df, metadata_dict)
    """
    set_seed(random_seed)
    df = model_data.copy()
    metadata = {
        "protected_attribute": protected_attribute,
        "bias_intensity": bias_intensity,
        "injection_method": injection_method,
        "random_seed": random_seed,
        "original_shape": df.shape,
        "before_stats": {},
        "after_stats": {}
    }

    if "hired" not in df.columns:
        raise ValueError("Dataset must contain 'hired' outcome column")

    # 2026-09-19 webapp-parity remediation: unknown/unsupported protected
    # attributes previously fell through every branch silently and returned
    # the ORIGINAL, unmodified dataframe with an empty-looking metadata dict
    # -- indistinguishable from "no bias found" rather than "not supported".
    # Mirrors the F1 fix (hard-fail on unknown columns) already applied to
    # the MCP server on 2026-09-18.
    if protected_attribute not in ("gender", "age"):
        raise ValueError(
            f"Unsupported protected_attribute '{protected_attribute}'. "
            f"This demo dataset/engine only supports bias injection for: gender, age. "
            f"(race/disability are selectable in the UI as attributes but have no "
            f"injection logic implemented yet -- selecting them must fail loudly, "
            f"not silently return unchanged data.)"
        )

    # Calculate before bias, and flag when a subgroup is too small for the
    # before/after hire-rate comparison to be statistically meaningful
    # (F2-equivalent guard).
    metadata["warnings"] = []
    if protected_attribute == "gender":
        male_mask = df["gender"] == "male"
        before_male_rate = df.loc[male_mask, "hired"].mean()
        before_female_rate = df.loc[~male_mask, "hired"].mean()
        metadata["before_stats"] = {"male_hire_rate": round(before_male_rate, 3), "female_hire_rate": round(before_female_rate, 3)}
        counts = {"male": int(male_mask.sum()), "female": int((~male_mask).sum())}
        metadata["subgroup_counts"] = counts
        if min(counts.values()) < MIN_SUBGROUP_SAMPLE_SIZE:
            metadata["warnings"].append(
                f"Insufficient data: subgroup counts {counts} below minimum "
                f"sample size ({MIN_SUBGROUP_SAMPLE_SIZE}) -- before/after "
                f"hire-rate comparison is not statistically reliable."
            )
    elif protected_attribute == "age":
        young_mask = df["age"] < 50
        before_young = df.loc[young_mask, "hired"].mean()
        before_old = df.loc[~young_mask, "hired"].mean()
        metadata["before_stats"] = {"under50_hire_rate": round(before_young, 3), "over50_hire_rate": round(before_old, 3)}
        counts = {"under50": int(young_mask.sum()), "over50": int((~young_mask).sum())}
        metadata["subgroup_counts"] = counts
        if min(counts.values()) < MIN_SUBGROUP_SAMPLE_SIZE:
            metadata["warnings"].append(
                f"Insufficient data: subgroup counts {counts} below minimum "
                f"sample size ({MIN_SUBGROUP_SAMPLE_SIZE}) -- before/after "
                f"hire-rate comparison is not statistically reliable."
            )

    # Apply injection
    if injection_method == "label_flip":
        # Flip positive outcomes for protected group
        if protected_attribute == "gender":
            protected_mask = df["gender"] == "female"
            flip_mask = protected_mask & (df["hired"] == 1)
            flip_count = int(flip_mask.sum() * bias_intensity)
            flip_indices = df[flip_mask].sample(n=flip_count, random_state=random_seed).index
            df.loc[flip_indices, "hired"] = 0
        elif protected_attribute == "age":
            protected_mask = df["age"] >= 50
            flip_mask = protected_mask & (df["hired"] == 1)
            flip_count = int(flip_mask.sum() * bias_intensity)
            flip_indices = df[flip_mask].sample(n=flip_count, random_state=random_seed).index
            df.loc[flip_indices, "hired"] = 0

    elif injection_method == "feature_skew":
        # Add noise to features correlated with protected attr
        if protected_attribute == "gender":
            protected_mask = df["gender"] == "female"
            noise = np.random.normal(0, bias_intensity * 15000, size=protected_mask.sum())
            df.loc[protected_mask, "previous_salary"] = (df.loc[protected_mask, "previous_salary"] + noise).astype(int)

    elif injection_method == "sampling_bias":
        # Undersample protected group
        if protected_attribute == "gender":
            protected_mask = df["gender"] == "female"
            keep_count = int(protected_mask.sum() * (1 - bias_intensity * 0.6))
            keep_indices = df[protected_mask].sample(n=keep_count, random_state=random_seed).index
            df = pd.concat([df[~protected_mask], df.loc[keep_indices]]).reset_index(drop=True)

    # Calculate after bias
    if protected_attribute == "gender":
        male_mask = df["gender"] == "male"
        after_male = df.loc[male_mask, "hired"].mean()
        after_female = df.loc[df["gender"] == "female", "hired"].mean()
        metadata["after_stats"] = {"male_hire_rate": round(after_male, 3), "female_hire_rate": round(after_female, 3)}
        bias_delta = abs(after_male - after_female) - abs(before_male_rate - before_female_rate)
        metadata["bias_amplification"] = round(max(0, bias_delta), 3)

    return df, metadata