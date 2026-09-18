"""
Core Fairness and Bias Audit Engine using Fairlearn and scikit-learn.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
from scipy.stats import chi2_contingency as _chi2_contingency
from ai_bias_lab.models import (
    FairnessMetrics, ComplianceStatus, MitigationRecommendation,
    MitigationType, FairnessMetricType, ProtectedAttribute,
    SubgroupCount, ConfidenceInterval, ProxyCorrelationResult
)

# FIX F2 (2026-09-18 remediation brief, P0): minimum subgroup sample size
# below which a fairness verdict is not statistically meaningful. A verdict
# on this few records is not distinguishable in shape from one backed by
# tens of thousands, so it must be flagged as INSUFFICIENT_DATA rather than
# COMPLIANT/WARNING/NON_COMPLIANT.
MIN_SUBGROUP_SAMPLE_SIZE = 30
DEFAULT_BOOTSTRAP_ITERATIONS = 1000


class FairnessAuditEngine:
    """
    Algorithmic Bias & Fairness Evaluation Engine.
    Implements EEOC 80% four-fifths rule, EU AI Act Article 10 metrics,
    and proxy-variable correlation detection.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def calculate_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        favorable_label: int = 1,
        bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
        random_seed: Optional[int] = None
    ) -> FairnessMetrics:
        """
        Calculates Statistical/Demographic Parity Difference, Equalized Odds Difference,
        Disparate Impact Ratio, and Bias Amplification Factor.

        FIX F2 (2026-09-18 remediation brief, P0): also computes per-subgroup
        counts, bootstrap 95% confidence intervals, and a warnings list, and
        forces the verdict to INSUFFICIENT_DATA (taking precedence over
        COMPLIANT/WARNING/NON_COMPLIANT) whenever any compared subgroup has
        fewer than MIN_SUBGROUP_SAMPLE_SIZE (30) records. A COMPLIANT verdict
        on n=8 must never look the same as one on n=50,000.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_features = np.asarray(sensitive_features)
        seed = random_seed if random_seed is not None else self.random_seed

        # Demographic / Statistical Parity Difference (selection rate difference)
        spd = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features))

        # Equalized Odds Difference (max disparity in TPR/FPR)
        eod = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features))

        # Disparate Impact Ratio: min(selection_rate) / max(selection_rate)
        df_eval = pd.DataFrame({"y_pred": y_pred, "sens": sensitive_features})
        selection_rates = df_eval.groupby("sens")["y_pred"].mean()
        subgroup_sizes = df_eval.groupby("sens")["y_pred"].count()
        subgroup_favorable = df_eval.groupby("sens")["y_pred"].sum()

        if len(selection_rates) >= 2 and selection_rates.max() > 0:
            di_ratio = float(selection_rates.min() / selection_rates.max())
        else:
            di_ratio = 1.0

        # Bias Amplification Factor: output disparity vs ground truth input disparity
        df_true = pd.DataFrame({"y_true": y_true, "sens": sensitive_features})
        true_rates = df_true.groupby("sens")["y_true"].mean()
        input_disp = float(abs(true_rates.max() - true_rates.min())) if len(true_rates) >= 2 else 0.01
        output_disp = float(abs(selection_rates.max() - selection_rates.min())) if len(selection_rates) >= 2 else 0.01
        amp_factor = round(output_disp / max(input_disp, 0.01), 3)

        # --- FIX F2: subgroup_counts (also satisfies F6's per-subgroup breakdown) ---
        subgroup_counts: Dict[str, SubgroupCount] = {}
        for grp in selection_rates.index:
            n = int(subgroup_sizes.loc[grp])
            n_fav = int(subgroup_favorable.loc[grp])
            subgroup_counts[str(grp)] = SubgroupCount(
                n=n, n_favorable=n_fav,
                selection_rate=round(float(selection_rates.loc[grp]), 4),
            )

        warnings: List[str] = []
        small_subgroups = [name for name, sc in subgroup_counts.items() if sc.n < MIN_SUBGROUP_SAMPLE_SIZE]
        insufficient_data = len(small_subgroups) > 0
        if insufficient_data:
            plural = "s" if len(small_subgroups) != 1 else ""
            details = ", ".join(f"{name} (n={subgroup_counts[name].n})" for name in small_subgroups)
            warnings.append(
                f"Insufficient data: subgroup{plural} below the minimum sample size of "
                f"{MIN_SUBGROUP_SAMPLE_SIZE}: {details}. Verdict is not statistically meaningful."
            )
        if len(subgroup_counts) < 2:
            warnings.append(
                "Single-category sensitive attribute: only one subgroup present, so no "
                "between-group comparison (disparate impact, statistical parity) is possible."
            )

        # --- FIX F2: bootstrap 95% confidence intervals ---
        di_ci = None
        spd_ci = None
        if len(selection_rates) >= 2:
            di_ci = self._bootstrap_ci(
                lambda yp, s: FairnessAuditEngine._disparate_impact_from_arrays(yp, s),
                y_pred, sensitive_features, bootstrap_iterations, seed,
            )
            spd_ci = self._bootstrap_ci(
                lambda yp, s: FairnessAuditEngine._statistical_parity_from_arrays(yp, s),
                y_pred, sensitive_features, bootstrap_iterations, seed,
            )
            if di_ci is not None and (di_ci.upper - di_ci.lower) > 0.30:
                warnings.append(
                    f"Wide 95% confidence interval on disparate_impact_ratio "
                    f"[{di_ci.lower}, {di_ci.upper}]: point estimate is not precise."
                )

        # Status classification based on the 80% / four-fifths rule and EU AI Act Art 10.
        # INSUFFICIENT_DATA takes precedence over every other verdict (FIX F2).
        if insufficient_data:
            di_status = ComplianceStatus.INSUFFICIENT_DATA
            art10_status = ComplianceStatus.INSUFFICIENT_DATA
        elif di_ratio >= 0.80 and abs(spd) <= 0.10:
            di_status = ComplianceStatus.COMPLIANT
            art10_status = ComplianceStatus.COMPLIANT
        elif di_ratio >= 0.65 or abs(spd) <= 0.20:
            di_status = ComplianceStatus.WARNING
            art10_status = ComplianceStatus.WARNING
        else:
            di_status = ComplianceStatus.NON_COMPLIANT
            art10_status = ComplianceStatus.NON_COMPLIANT

        return FairnessMetrics(
            statistical_parity_difference=round(spd, 4),
            equalized_odds_difference=round(eod, 4),
            disparate_impact_ratio=round(di_ratio, 4),
            bias_amplification_factor=amp_factor,
            disparate_impact_status=di_status,
            eu_ai_act_art10_status=art10_status,
            subgroup_counts=subgroup_counts,
            disparate_impact_ci_95=di_ci,
            statistical_parity_ci_95=spd_ci,
            warnings=warnings,
        )

    @staticmethod
    def _disparate_impact_from_arrays(y_pred: np.ndarray, sens: np.ndarray) -> float:
        df = pd.DataFrame({"y_pred": y_pred, "sens": sens})
        rates = df.groupby("sens")["y_pred"].mean()
        if len(rates) >= 2 and rates.max() > 0:
            return float(rates.min() / rates.max())
        return 1.0

    @staticmethod
    def _statistical_parity_from_arrays(y_pred: np.ndarray, sens: np.ndarray) -> float:
        df = pd.DataFrame({"y_pred": y_pred, "sens": sens})
        rates = df.groupby("sens")["y_pred"].mean()
        if len(rates) >= 2:
            return float(rates.max() - rates.min())
        return 0.0

    def _bootstrap_ci(
        self,
        stat_fn,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        n_resamples: int,
        seed: int,
    ) -> Optional[ConfidenceInterval]:
        """FIX F2: bootstrap percentile 95% CI for a point-estimate statistic,
        resampling rows with replacement, stratified nowhere (plain i.i.d.
        resample of the paired (y_pred, sensitive) rows) -- deterministic
        given `seed`, so results are reproducible for the same seed (see F8
        random_seed echo)."""
        n = len(y_pred)
        if n < 2:
            return None
        rng = np.random.default_rng(seed)
        stats = np.empty(n_resamples, dtype=float)
        idx_range = np.arange(n)
        for i in range(n_resamples):
            sample_idx = rng.choice(idx_range, size=n, replace=True)
            stats[i] = stat_fn(y_pred[sample_idx], sensitive_features[sample_idx])
        lower = float(np.percentile(stats, 2.5))
        upper = float(np.percentile(stats, 97.5))
        return ConfidenceInterval(
            lower=round(lower, 4), upper=round(upper, 4),
            method="bootstrap_percentile", n_resamples=n_resamples,
        )

    def detect_proxy_correlations(
        self,
        df: pd.DataFrame,
        sensitive_column: str,
        threshold: float = 0.35
    ) -> ProxyCorrelationResult:
        """
        Detects proxy features in the dataset that correlate strongly with
        the sensitive attribute (e.g. postcode correlating with ethnicity or age).

        FIX F4 (2026-09-18 remediation brief, P1): previously returned a bare
        Dict[str, float] and used ONLY Pearson correlation on
        category-code-encoded columns. This meant a genuinely categorical
        sensitive attribute (gender, postcode_cluster) would frequently
        produce an empty {} not because no proxy exists, but because
        Pearson correlation on arbitrary category-code integers is not a
        meaningful association measure for nominal categories -- "no proxy
        found" and "not meaningfully computed" looked identical. This was
        confirmed against the built-in dataset: proxy detection worked for
        `age` (numeric, found pension_accrual_years at 0.98) but returned
        {} for `gender` and `postcode_cluster` (categorical).

        Now: numeric sensitive columns still use Pearson (unchanged
        behaviour/values, see T6 regression guard). Categorical sensitive
        columns use Cramér's V against other categorical columns and a
        correlation-ratio (eta) style association for numeric columns,
        instead of being silently skipped. The result always states
        status='computed' with a method, or status='not_computed' with an
        explicit reason -- never a bare, ambiguous {}.
        """
        if sensitive_column not in df.columns:
            return ProxyCorrelationResult(status="not_computed", reason=f"sensitive_column '{sensitive_column}' not found in dataset")

        sens_col_data = df[sensitive_column]
        sens_is_numeric = np.issubdtype(sens_col_data.dtype, np.number)

        correlations: Dict[str, float] = {}
        method_used = "pearson" if sens_is_numeric else "cramers_v_and_correlation_ratio"

        if sens_is_numeric:
            for col in df.columns:
                if col == sensitive_column:
                    continue
                col_series = df[col]
                col_numeric = col_series if np.issubdtype(col_series.dtype, np.number) else col_series.astype("category").cat.codes
                try:
                    corr = abs(float(pd.Series(sens_col_data).corr(pd.Series(col_numeric))))
                    if np.isfinite(corr) and corr >= threshold:
                        correlations[col] = round(corr, 3)
                except Exception:
                    continue
        else:
            # Categorical sensitive attribute: use Cramer's V against other
            # categorical columns, and the correlation ratio (eta) against
            # numeric columns -- both are bounded [0, 1] like Pearson |r|,
            # so the same `threshold` remains comparable.
            for col in df.columns:
                if col == sensitive_column:
                    continue
                col_series = df[col]
                try:
                    if np.issubdtype(col_series.dtype, np.number):
                        assoc = self._correlation_ratio(sens_col_data, col_series)
                    else:
                        assoc = self._cramers_v(sens_col_data, col_series)
                    if np.isfinite(assoc) and assoc >= threshold:
                        correlations[col] = round(float(assoc), 3)
                except Exception:
                    continue

        return ProxyCorrelationResult(status="computed", correlations=correlations, method=method_used)

    @staticmethod
    def _cramers_v(a: pd.Series, b: pd.Series) -> float:
        """Cramer's V association measure between two categorical series
        (0 = no association, 1 = perfect association), bias-corrected per
        Bergsma (2013)."""
        confusion = pd.crosstab(a, b)
        chi2 = float(_chi2_contingency(confusion.values)[0])
        n = confusion.values.sum()
        if n == 0:
            return 0.0
        phi2 = chi2 / n
        r, k = confusion.shape
        phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
        r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
        k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
        denom = min(k_corr - 1, r_corr - 1)
        if denom <= 0:
            return 0.0
        return float(np.sqrt(phi2_corr / denom))

    @staticmethod
    def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
        """Correlation ratio (eta) between a categorical series and a numeric
        series: proportion of the numeric variable's variance explained by
        category membership (0 = none, 1 = fully explained)."""
        values = pd.Series(values).astype(float)
        categories = pd.Series(categories)
        overall_mean = values.mean()
        ss_total = float(((values - overall_mean) ** 2).sum())
        if ss_total == 0:
            return 0.0
        ss_between = 0.0
        for _, group in values.groupby(categories):
            ss_between += len(group) * (group.mean() - overall_mean) ** 2
        return float(np.sqrt(ss_between / ss_total))

    def suggest_mitigations(self, metrics: FairnessMetrics) -> List[MitigationRecommendation]:
        """
        Recommends concrete technical mitigations based on identified bias gaps.
        """
        recommendations = []

        # FIX F3 (2026-09-18 remediation brief, P1): expected_disparate_impact_improvement
        # was a fixed constant (0.15 / 0.12) presented as a data-derived
        # estimate, identical regardless of how severe the actual disparity
        # was. Replaced with heuristic_target_gap (target_value minus
        # current_value -- a plain arithmetic gap, not a claim about what
        # will happen) plus an explicit `basis` field disclosing it is not
        # estimated from the dataset. Only post_processing's gap is
        # naturally data-dependent (target minus current); pre/in-processing
        # gaps are also now computed the same honest way rather than left as
        # unrelated fixed numbers.
        if metrics.disparate_impact_ratio < 0.80:
            recommendations.append(MitigationRecommendation(
                mitigation_type=MitigationType.POST_PROCESSING_THRESHOLD,
                target_metric=FairnessMetricType.DISPARATE_IMPACT,
                current_value=metrics.disparate_impact_ratio,
                target_value=0.85,
                recommended_action=(
                    "Implement Fairlearn ThresholdOptimizer to establish subgroup-specific decision "
                    "thresholds, ensuring selection rates satisfy the 80% four-fifths rule under EU AI Act Art. 10."
                ),
                heuristic_target_gap=round(0.85 - metrics.disparate_impact_ratio, 3),
            ))

        if abs(metrics.statistical_parity_difference) > 0.10:
            recommendations.append(MitigationRecommendation(
                mitigation_type=MitigationType.PRE_PROCESSING_REWEIGHT,
                target_metric=FairnessMetricType.STATISTICAL_PARITY,
                current_value=metrics.statistical_parity_difference,
                target_value=0.05,
                recommended_action=(
                    "Apply sample reweighting (Fairlearn CorrelationRemover / sample weights) to balance "
                    "representation of underrepresented demographic cohorts in training data."
                ),
                heuristic_target_gap=round(abs(metrics.statistical_parity_difference) - 0.05, 3),
            ))

        if metrics.equalized_odds_difference > 0.15:
            recommendations.append(MitigationRecommendation(
                mitigation_type=MitigationType.IN_PROCESSING_EXPONENTIATED,
                target_metric=FairnessMetricType.EQUALIZED_ODDS,
                current_value=metrics.equalized_odds_difference,
                target_value=0.08,
                recommended_action=(
                    "Retrain model using Fairlearn ExponentiatedGradient with EqualizedOdds constraint "
                    "to equalize False Positive and False Negative rates across protected classes."
                ),
                heuristic_target_gap=round(metrics.equalized_odds_difference - 0.08, 3),
            ))

        return recommendations
