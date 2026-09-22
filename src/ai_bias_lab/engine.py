"""
Core Fairness and Bias Audit Engine using Fairlearn and scikit-learn.
"""

from typing import Dict, List, Any, Optional, Tuple
import hashlib
import json as _json
from datetime import datetime, timezone
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

# FIX F8 (2026-09-18 remediation brief, P2): server version string, echoed
# into every audit response's tool_version field.
# FIX N5 (round 2 verification, 2026-09-22): bumped to 1.2.0 -- this
# release fixes O1 (LLM prompt bias ignored), N5 itself (null audit trail /
# non-reproducible bootstrap), N2 (one-sided SPD CI), N7 (fabricated 1.0
# equalized odds), N1 (reference==comparison on ties), N4
# (bias_amplification_factor sentinel/undefined behaviour), and N6 (stray
# Art. 10 attribution in mitigation text).
TOOL_VERSION = "1.2.0"

# FIX N4 (round 2 verification, 2026-09-22): below this ground-truth
# selection-rate disparity, bias_amplification_factor's denominator is so
# close to zero that the ratio becomes numerically explosive and stops
# being an interpretable measurement (this is what produced the 100.0
# sentinel-looking value on Probe B: input_disp=0.01 floor, output_disp=1.0
# -> 100.0, versus 1.0 on Probe D with identical headline metrics). Below
# this floor the factor is reported as null with a warning rather than a
# number that looks precise but isn't.
BIAS_AMPLIFICATION_MIN_INPUT_DISPARITY = 0.05


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """FIX F8: SHA-256 of the canonicalised (sorted columns, fixed row
    order, JSON-serialised) input records, so a result can be tied back to
    the exact data version it was computed against. Stable for the
    built-in benchmark dataset (same seed -> same fingerprint)."""
    canonical = df[sorted(df.columns)].to_dict(orient="records")
    canonical_json = _json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


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
        random_seed: Optional[int] = None,
        reference_group: Optional[str] = None,
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

        FIX F6 (2026-09-18 remediation brief, P1): disparate_impact_ratio and
        statistical_parity_difference are pairwise (min-selection-rate group
        vs max-selection-rate group), which was never stated when
        sensitive_features has more than 2 categories (e.g.
        postcode_cluster). `reference_group` param lets the caller override
        which subgroup is treated as the baseline; when omitted, the
        subgroup with the highest selection rate is used (documented, not
        silent).

        FIX N1 (round 2 verification, 2026-09-22): on a tie for highest
        selection rate (or a single-subgroup input), reference_group and
        comparison_group previously could resolve to the SAME subgroup
        (pandas idxmax/idxmin both return the first-occurring tied label,
        so a self-comparison silently reported DI=1.0 / SPD=0.0 "by
        construction" rather than by measurement). A deterministic tiebreak
        is now applied -- see `_pick_reference_and_comparison` -- and
        comparison_group is always chosen from the subgroups REMAINING
        after reference_group is removed, so the two are never equal when
        2+ subgroups exist. With genuinely one subgroup, comparison_group
        is None and reference_group_selection says so explicitly.

        FIX N2 (round 2 verification, 2026-09-22): statistical_parity_ci_95
        is bootstrapped on the SIGNED statistic (reference_group's rate
        minus comparison_group's rate, for the SAME fixed reference/
        comparison pair on every resample), not on an implicitly clipped or
        absolute-valued quantity. The interval can legitimately straddle
        zero, and its `excludes_zero` flag says whether the direction of
        the disparity survives resampling noise at the 95% level.
        `signed_statistical_parity_difference` carries the signed point
        estimate; `statistical_parity_difference` remains the unsigned
        magnitude for backward compatibility.

        FIX N7 (round 2 verification, 2026-09-22): equalized_odds_difference
        is computed from explicit per-subgroup TPR/FPR rather than taken
        from fairlearn's equalized_odds_difference() directly, because that
        function silently substitutes 0 for an undefined rate (a subgroup
        with no positive, or no negative, ground-truth cases -- observed to
        fabricate a maximal 1.0 on a 2-row probe). When any compared
        subgroup has an undefined TPR or FPR, this method returns
        equalized_odds_difference=None with a warning naming the subgroup
        and which rate was undefined, instead of a fabricated number.

        FIX N4 (round 2 verification, 2026-09-22): bias_amplification_factor
        is None when the ground-truth (input) selection-rate disparity is
        below BIAS_AMPLIFICATION_MIN_INPUT_DISPARITY -- in that regime the
        ratio is numerically explosive (a near-zero denominator) and not an
        interpretable measurement, which is what previously produced a
        100.0 value on one probe and 1.0 on another with otherwise
        identical headline metrics.

        FIX N5 (round 2 verification, 2026-09-22): audit_timestamp,
        tool_version, dataset_fingerprint and random_seed_used are always
        populated on the returned FairnessMetrics (previously left at their
        null defaults on this code path -- only the MCP dataset-audit path
        set them). dataset_fingerprint hashes the (y_true, y_pred,
        sensitive_features) triple actually evaluated. random_seed_used
        always reflects the seed that drove the bootstrap resampling, so
        the same call with the same seed reproduces byte-identical CIs.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_features = np.asarray(sensitive_features)
        seed = random_seed if random_seed is not None else self.random_seed
        # Stringify subgroup labels up front (once), so every downstream
        # groupby index, .loc[] lookup, and returned reference/comparison
        # name uses the SAME (string) key consistently -- avoids a KeyError
        # when sensitive_features is numeric (e.g. integer age values),
        # where reference_group/comparison_group names are always str().
        sensitive_features_str = np.asarray([str(s) for s in sensitive_features])

        # Equalized Odds Difference (max disparity in TPR/FPR).
        # FIX N7: computed from explicit per-subgroup TPR/FPR (see
        # _equalized_odds_from_arrays) instead of fairlearn's
        # equalized_odds_difference(), which silently treats an undefined
        # per-subgroup rate as 0.
        eod, eod_warnings = FairnessAuditEngine._equalized_odds_from_arrays(y_true, y_pred, sensitive_features_str)

        # Disparate Impact Ratio: min(selection_rate) / max(selection_rate)
        df_eval = pd.DataFrame({"y_pred": y_pred, "sens": sensitive_features_str})
        selection_rates = df_eval.groupby("sens")["y_pred"].mean()
        subgroup_sizes = df_eval.groupby("sens")["y_pred"].count()
        subgroup_favorable = df_eval.groupby("sens")["y_pred"].sum()

        if len(selection_rates) >= 2 and selection_rates.max() > 0:
            di_ratio = float(selection_rates.min() / selection_rates.max())
        else:
            di_ratio = 1.0

        # Bias Amplification Factor: output disparity vs ground truth input
        # disparity. FIX N4: None (not a sentinel number) when the
        # ground-truth disparity is too close to zero for the ratio to be a
        # meaningful measurement -- see BIAS_AMPLIFICATION_MIN_INPUT_DISPARITY.
        df_true = pd.DataFrame({"y_true": y_true, "sens": sensitive_features_str})
        true_rates = df_true.groupby("sens")["y_true"].mean()
        input_disp = float(abs(true_rates.max() - true_rates.min())) if len(true_rates) >= 2 else 0.0
        output_disp = float(abs(selection_rates.max() - selection_rates.min())) if len(selection_rates) >= 2 else 0.0
        amp_undefined = input_disp < BIAS_AMPLIFICATION_MIN_INPUT_DISPARITY
        amp_factor: Optional[float] = None if amp_undefined else round(output_disp / input_disp, 3)

        # --- FIX F2: subgroup_counts (also satisfies F6's per-subgroup breakdown) ---
        subgroup_counts: Dict[str, SubgroupCount] = {}
        for grp in selection_rates.index:
            n = int(subgroup_sizes.loc[grp])
            n_fav = int(subgroup_favorable.loc[grp])
            subgroup_counts[str(grp)] = SubgroupCount(
                n=n, n_favorable=n_fav,
                selection_rate=round(float(selection_rates.loc[grp]), 4),
            )

        # --- FIX F6 / FIX N1: reference_group / comparison_group ---
        # disparate_impact_ratio and statistical_parity_difference are
        # pairwise: min-selection-rate group vs max-selection-rate group.
        # FIX N1: ties are broken deterministically (largest n, then
        # lexical order on the group label) and comparison_group is always
        # chosen from the subgroups REMAINING after reference_group is
        # removed, so the two are never equal when 2+ subgroups exist.
        ref_group_name, cmp_group_name, ref_selection_rule = FairnessAuditEngine._pick_reference_and_comparison(
            selection_rates, subgroup_sizes, reference_group,
        )

        warnings: List[str] = []
        warnings.extend(eod_warnings)
        if amp_undefined and len(true_rates) >= 1:
            warnings.append(
                f"bias_amplification_factor not computed: ground-truth (y_true) selection-rate "
                f"disparity across subgroups is {round(input_disp, 4)}, below the "
                f"{BIAS_AMPLIFICATION_MIN_INPUT_DISPARITY} minimum needed for the ratio to be a "
                f"meaningful measurement rather than a near-zero-denominator artefact."
            )
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

        # --- FIX F2 / FIX N2: bootstrap 95% confidence intervals ---
        # di_ci remains an unsigned ratio CI (bounded [0, 1], no direction
        # to lose -- excludes_zero is left None, see ConfidenceInterval).
        # spd_ci (FIX N2) is bootstrapped on the SIGNED statistical parity
        # statistic -- reference_group's rate minus comparison_group's
        # rate, using the SAME fixed reference/comparison subgroup labels
        # for every resample -- so a resample where the comparison group
        # happens to have the higher rate correctly contributes a NEGATIVE
        # value, and the interval can straddle zero.
        di_ci = None
        spd_ci = None
        signed_spd_point: Optional[float] = None
        if len(selection_rates) >= 2 and cmp_group_name is not None:
            signed_spd_point = float(selection_rates.loc[ref_group_name] - selection_rates.loc[cmp_group_name])
            di_ci = self._bootstrap_ci(
                lambda yp, s: FairnessAuditEngine._disparate_impact_from_arrays(yp, s),
                y_pred, sensitive_features_str, bootstrap_iterations, seed,
            )
            spd_ci = self._bootstrap_ci(
                lambda yp, s: FairnessAuditEngine._signed_statistical_parity_from_arrays(yp, s, ref_group_name, cmp_group_name),
                y_pred, sensitive_features_str, bootstrap_iterations, seed,
                signed=True,
            )
            if di_ci is not None and (di_ci.upper - di_ci.lower) > 0.30:
                warnings.append(
                    f"Wide 95% confidence interval on disparate_impact_ratio "
                    f"[{di_ci.lower}, {di_ci.upper}]: point estimate is not precise."
                )
            if spd_ci is not None and spd_ci.excludes_zero is False:
                warnings.append(
                    f"95% confidence interval on statistical parity "
                    f"[{spd_ci.lower}, {spd_ci.upper}] includes 0: the direction of this disparity "
                    f"is not distinguishable from sampling noise at the 95% level."
                )

        spd = abs(signed_spd_point) if signed_spd_point is not None else 0.0

        # Status classification based on the 80% / four-fifths rule.
        # FIX F5 (2026-09-18 remediation brief, P1): this threshold is the
        # US EEOC four-fifths rule (29 CFR 1607.4(D)), NOT an EU AI Act
        # Article 10 verdict -- renamed the variable and output field
        # accordingly. INSUFFICIENT_DATA still takes precedence (FIX F2).
        if insufficient_data:
            di_status = ComplianceStatus.INSUFFICIENT_DATA
            eeoc_status = ComplianceStatus.INSUFFICIENT_DATA
        elif di_ratio >= 0.80 and abs(spd) <= 0.10:
            di_status = ComplianceStatus.COMPLIANT
            eeoc_status = ComplianceStatus.COMPLIANT
        elif di_ratio >= 0.65 or abs(spd) <= 0.20:
            di_status = ComplianceStatus.WARNING
            eeoc_status = ComplianceStatus.WARNING
        else:
            di_status = ComplianceStatus.NON_COMPLIANT
            eeoc_status = ComplianceStatus.NON_COMPLIANT

        # FIX N5: audit trail -- always populated, never left null.
        fingerprint = FairnessAuditEngine._fingerprint_arrays(y_true, y_pred, sensitive_features)

        return FairnessMetrics(
            statistical_parity_difference=round(spd, 4),
            signed_statistical_parity_difference=round(signed_spd_point, 4) if signed_spd_point is not None else None,
            equalized_odds_difference=round(eod, 4) if eod is not None else None,
            disparate_impact_ratio=round(di_ratio, 4),
            bias_amplification_factor=amp_factor,
            disparate_impact_status=di_status,
            eeoc_four_fifths_status=eeoc_status,
            # eu_ai_act_art10_status is auto-synced to eeoc_four_fifths_status
            # by the model_validator on FairnessMetrics (deprecated alias).
            # art10_documentation_status is left at its not_assessed default.
            subgroup_counts=subgroup_counts,
            disparate_impact_ci_95=di_ci,
            statistical_parity_ci_95=spd_ci,
            warnings=warnings,
            reference_group=ref_group_name,
            comparison_group=cmp_group_name,
            reference_group_selection=ref_selection_rule,
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
            tool_version=TOOL_VERSION,
            dataset_fingerprint=fingerprint,
            random_seed_used=seed,
        )

    @staticmethod
    def _pick_reference_and_comparison(
        selection_rates: "pd.Series",
        subgroup_sizes: "pd.Series",
        reference_group: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """FIX N1: deterministic reference/comparison subgroup selection
        that never returns the same subgroup for both when 2+ subgroups
        exist. Ties on selection rate are broken by: (1) largest subgroup
        n, then (2) lexically smallest label -- both fully deterministic,
        so the same input always yields the same choice. comparison_group
        is always chosen from the subgroups remaining after reference_group
        is excluded, never from the full set.
        """
        labels = [str(g) for g in selection_rates.index]
        if len(labels) == 0:
            return None, None, None
        if len(labels) == 1:
            return labels[0], None, "undefined: only one subgroup present"

        def _tie_break_pick(rates: "pd.Series", pick_max: bool) -> str:
            target = rates.max() if pick_max else rates.min()
            # Keep the ORIGINAL (possibly non-str, e.g. int age) index
            # labels for .loc lookups against subgroup_sizes, which shares
            # that same original index -- only stringify for the final
            # returned group name and the lexical tiebreak key.
            tied_orig = [g for g in rates.index if rates.loc[g] == target]
            if len(tied_orig) == 1:
                return str(tied_orig[0])
            tied_orig.sort(key=lambda g: (-int(subgroup_sizes.loc[g]), str(g)))
            return str(tied_orig[0])

        if reference_group is not None:
            if str(reference_group) not in labels:
                raise ValueError(
                    f"reference_group '{reference_group}' not found among subgroups: {labels}"
                )
            ref_group_name = str(reference_group)
            remaining = selection_rates[[str(g) != ref_group_name for g in selection_rates.index]]
            cmp_group_name = _tie_break_pick(remaining, pick_max=False)
            return ref_group_name, cmp_group_name, "caller-specified via reference_group parameter"

        top_rate = selection_rates.max()
        tied_top = [str(g) for g in selection_rates.index if selection_rates.loc[g] == top_rate]
        ref_group_name = _tie_break_pick(selection_rates, pick_max=True)
        remaining = selection_rates[[str(g) != ref_group_name for g in selection_rates.index]]
        cmp_group_name = _tie_break_pick(remaining, pick_max=False)
        if len(tied_top) > 1:
            ref_selection_rule = "highest selection rate (tie broken by largest n, then lexical order)"
        else:
            ref_selection_rule = "highest selection rate"
        return ref_group_name, cmp_group_name, ref_selection_rule

    @staticmethod
    def _equalized_odds_from_arrays(
        y_true: np.ndarray, y_pred: np.ndarray, sens: np.ndarray
    ) -> Tuple[Optional[float], List[str]]:
        """FIX N7: per-subgroup TPR/FPR computed explicitly so an undefined
        rate (a subgroup with zero positive -- or zero negative --
        ground-truth cases) is detected and reported, instead of being
        silently treated as 0 by fairlearn's equalized_odds_difference
        (which is what produced a fabricated 1.0 on a probe where one
        subgroup had no positive ground-truth cases at all).

        Returns (equalized_odds_difference_or_None, warnings).
        """
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "sens": sens})
        groups = sorted(df["sens"].unique().tolist(), key=str)
        if len(groups) < 2:
            return None, []

        tprs: Dict[str, Optional[float]] = {}
        fprs: Dict[str, Optional[float]] = {}
        undefined_notes: List[str] = []
        for g in groups:
            sub = df[df["sens"] == g]
            positives = sub[sub["y_true"] == 1]
            negatives = sub[sub["y_true"] == 0]
            if len(positives) == 0:
                tprs[str(g)] = None
                undefined_notes.append(f"{g} (no positive ground-truth cases, TPR undefined)")
            else:
                tprs[str(g)] = float((positives["y_pred"] == 1).mean())
            if len(negatives) == 0:
                fprs[str(g)] = None
                undefined_notes.append(f"{g} (no negative ground-truth cases, FPR undefined)")
            else:
                fprs[str(g)] = float((negatives["y_pred"] == 1).mean())

        if undefined_notes:
            warning = (
                "equalized_odds_difference not computed: undefined TPR or FPR in subgroup(s) "
                + "; ".join(undefined_notes)
                + ". An undefined rate cannot be safely compared to a defined one without "
                "fabricating a value."
            )
            return None, [warning]

        defined_tprs = [v for v in tprs.values() if v is not None]
        defined_fprs = [v for v in fprs.values() if v is not None]
        tpr_diff = max(defined_tprs) - min(defined_tprs)
        fpr_diff = max(defined_fprs) - min(defined_fprs)
        return float(max(tpr_diff, fpr_diff)), []

    @staticmethod
    def _disparate_impact_from_arrays(y_pred: np.ndarray, sens: np.ndarray) -> float:
        df = pd.DataFrame({"y_pred": y_pred, "sens": sens})
        rates = df.groupby("sens")["y_pred"].mean()
        if len(rates) >= 2 and rates.max() > 0:
            return float(rates.min() / rates.max())
        return 1.0

    @staticmethod
    def _signed_statistical_parity_from_arrays(
        y_pred: np.ndarray, sens: np.ndarray, ref_group: str, cmp_group: str
    ) -> float:
        """FIX N2: signed statistical parity for a FIXED (reference,
        comparison) pair of subgroup labels -- reference_group's selection
        rate minus comparison_group's selection rate, for this resample.
        Unlike an argmax/argmin-based statistic (unsigned by construction,
        since it always subtracts the smaller from the larger), this CAN be
        negative when a resample happens to give the comparison group the
        higher rate -- exactly the behaviour a bootstrap CI needs to show
        the true, potentially zero-straddling sampling distribution. If a
        label is entirely absent from a resample (possible only at very
        small n), its rate is treated as 0.0 for that draw -- a documented
        floor, not a silent NaN propagating into the percentile calculation.
        """
        df = pd.DataFrame({"y_pred": y_pred, "sens": sens})
        rates = df.groupby("sens")["y_pred"].mean()
        ref_rate = float(rates.loc[ref_group]) if ref_group in rates.index else 0.0
        cmp_rate = float(rates.loc[cmp_group]) if cmp_group in rates.index else 0.0
        return ref_rate - cmp_rate

    @staticmethod
    def _fingerprint_arrays(y_true: np.ndarray, y_pred: np.ndarray, sens: np.ndarray) -> str:
        """FIX N5: SHA-256 of the (y_true, y_pred, sensitive_features)
        triple actually evaluated, so calculate_fairness_metrics' output
        can be tied back to the exact inputs it was computed against, the
        same way compute_dataset_fingerprint does for full-dataset audits.
        """
        canonical = {
            "y_true": np.asarray(y_true).tolist(),
            "y_pred": np.asarray(y_pred).tolist(),
            "sensitive_features": [str(s) for s in np.asarray(sens).tolist()],
        }
        canonical_json = _json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def _bootstrap_ci(
        self,
        stat_fn,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        n_resamples: int,
        seed: int,
        signed: bool = False,
    ) -> Optional[ConfidenceInterval]:
        """FIX F2: bootstrap percentile 95% CI for a point-estimate statistic,
        resampling rows with replacement, stratified nowhere (plain i.i.d.
        resample of the paired (y_pred, sensitive) rows) -- deterministic
        given `seed`, so results are reproducible for the same seed (see F8/
        N5 random_seed echo).

        FIX N2: `signed=True` marks this CI as being on a signed statistic
        (can legitimately be negative); in that case `excludes_zero` is set
        to whether the resulting interval contains 0. For unsigned
        statistics (signed=False, e.g. the disparate impact ratio, which is
        bounded in [0, 1]), `excludes_zero` is left None -- "does this
        interval exclude zero" is not a meaningful question for a ratio
        whose ideal value is 1.0.
        """
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
        excludes_zero = None
        if signed:
            excludes_zero = bool(lower > 0.0 or upper < 0.0)
        return ConfidenceInterval(
            lower=round(lower, 4), upper=round(upper, 4),
            method="bootstrap_percentile", n_resamples=n_resamples,
            excludes_zero=excludes_zero,
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

        FIX F4 (2026-09-18 remediation brief, P1) / FIX O5 (round 2
        verification, 2026-09-22): previously returned a bare Dict[str,
        float] and used ONLY Pearson correlation on category-code-encoded
        columns. This meant a genuinely categorical sensitive attribute
        (gender, postcode_cluster) would frequently produce an empty {} not
        because no proxy exists, but because Pearson correlation on
        arbitrary category-code integers is not a meaningful association
        measure for nominal categories -- "no proxy found" and "not
        meaningfully computed" looked identical. Confirmed against the
        built-in dataset: proxy detection worked for `age` (numeric, found
        pension_accrual_years at 0.98) but returned {} for `gender` and
        `postcode_cluster` (categorical).

        Fix (O5): numeric sensitive columns still use Pearson (unchanged
        behaviour/values, regression-guarded). Categorical sensitive
        columns now use Cramer's V against other categorical columns and a
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
                    # FIX N6 (round 2 verification, 2026-09-22): this string
                    # still attributed the 80% four-fifths rule to "EU AI
                    # Act Art. 10" after the field itself was correctly
                    # renamed to eeoc_four_fifths_status (FIX F5). The rule
                    # originates in the US EEOC Uniform Guidelines (29 CFR
                    # 1607.4(D)), not EU AI Act Article 10 (which covers
                    # data governance and defines no numeric threshold).
                    "Implement Fairlearn ThresholdOptimizer to establish subgroup-specific decision "
                    "thresholds, ensuring selection rates satisfy the US EEOC 80% four-fifths rule "
                    "(29 CFR 1607.4(D))."
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

        # FIX N7: equalized_odds_difference may now be None (not computable
        # -- see calculate_fairness_metrics). Guard the comparison so a
        # None value does not raise, and skip this recommendation (silence
        # is correct here: we cannot recommend against an undefined number,
        # and the None itself plus its warning already flags the gap in
        # the underlying data to the caller).
        if metrics.equalized_odds_difference is not None and metrics.equalized_odds_difference > 0.15:
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
