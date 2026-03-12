"""
enterprise_eda.py
=================
Production-grade Exploratory Data Analysis (EDA) module for enterprise analytics pipelines.

This module assumes the input DataFrame has already been:
  - Cleaned
  - Type-corrected
  - Deduplicated
  - Profiled for missing values

Its sole responsibility is deep EDA to power downstream visualization and reporting systems.

Author  : Senior Data Engineering Team
Version : 1.0.0
"""
from __future__ import annotations
from utils.state import AgentState

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)


# ===========================================================================
# CONSTANTS
# ===========================================================================
STRONG_CORR_THRESHOLD: float = 0.6
HIGH_CARDINALITY_THRESHOLD: int = 50
SKEWNESS_THRESHOLD: float = 1.0
KURTOSIS_THRESHOLD: float = 3.0
IQR_OUTLIER_FENCE: float = 1.5
LOG_TRANSFORM_SKEW_THRESHOLD: float = 1.5
PERCENTILES: List[float] = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
MAX_CONTINGENCY_CARDINALITY: int = 20  # skip contingency tables for very high-cardinality cols
VIF_THRESHOLD: float = 5.0             # Variance Inflation Factor threshold


# ===========================================================================
# INPUT VALIDATION
# ===========================================================================

def _validate_input(df: pd.DataFrame) -> None:
    """
    Validate that the input is a non-empty pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If ``df`` is empty (no rows or no columns).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__} instead."
        )
    if df.empty:
        raise ValueError("The input DataFrame is empty (0 rows or 0 columns).")
    logger.info("Input validation passed — shape: %s", df.shape)


# ===========================================================================
# HELPER 1: FEATURE TYPE DETECTION
# ===========================================================================

def detect_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorise every column into one of four semantic groups:
    numeric, categorical, datetime, or boolean.

    Logic
    -----
    - ``bool`` dtype          → boolean
    - ``datetime`` dtype      → datetime
    - ``object`` / ``category`` dtype → categorical
    - All remaining numeric dtypes   → numeric

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Keys: ``"numeric"``, ``"categorical"``, ``"datetime"``, ``"boolean"``
        Values: sorted lists of column names.
    """
    feature_types: Dict[str, List[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
    }

    for col in df.columns:
        dtype = df[col].dtype

        if dtype == bool or str(dtype) == "boolean":
            feature_types["boolean"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            feature_types["datetime"].append(col)
        elif isinstance(dtype, pd.CategoricalDtype) or dtype == object:
            feature_types["categorical"].append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            feature_types["numeric"].append(col)
        else:
            # Fallback — treat as categorical
            feature_types["categorical"].append(col)

    for k, v in feature_types.items():
        logger.info("Feature type '%s': %d columns", k, len(v))

    return feature_types


# ===========================================================================
# HELPER 2: NUMERIC UNIVARIATE ANALYSIS
# ===========================================================================

def numeric_univariate_analysis(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> Dict[str, Any]:
    """
    Compute detailed distribution statistics for every numeric feature.

    Metrics computed per column
    ---------------------------
    - count, mean, median, std, variance
    - min, max, range
    - skewness, kurtosis
    - percentiles (1st through 99th)
    - coefficient of variation

    Parameters
    ----------
    df          : pd.DataFrame
    numeric_cols: list of numeric column names

    Returns
    -------
    dict  {column_name → stats_dict}
    """
    if not numeric_cols:
        logger.warning("No numeric columns found — skipping numeric univariate analysis.")
        return {}

    results: Dict[str, Any] = {}
    subset = df[numeric_cols]

    # Vectorised bulk statistics
    desc = subset.describe(percentiles=PERCENTILES).T  # shape: (n_cols, n_stats)

    for col in numeric_cols:
        series = subset[col].dropna()
        if series.empty:
            results[col] = {"error": "all_null"}
            continue

        col_stats = desc.loc[col].to_dict()

        # Scipy-based higher-order moments
        col_stats["skewness"] = float(stats.skew(series))
        col_stats["kurtosis"] = float(stats.kurtosis(series))  # excess kurtosis
        col_stats["median"] = float(series.median())
        col_stats["variance"] = float(series.var())
        col_stats["range"] = float(series.max() - series.min())
        col_stats["cv"] = (
            float(series.std() / series.mean()) if series.mean() != 0 else np.nan
        )
        col_stats["null_count"] = int(df[col].isna().sum())
        col_stats["null_pct"] = round(df[col].isna().mean() * 100, 4)

        results[col] = col_stats

    logger.info("Numeric univariate analysis complete for %d columns.", len(results))
    return results


# ===========================================================================
# HELPER 3: CATEGORICAL UNIVARIATE ANALYSIS
# ===========================================================================

def categorical_univariate_analysis(
    df: pd.DataFrame,
    categorical_cols: List[str],
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Compute frequency distributions and cardinality metrics for categorical features.

    Parameters
    ----------
    df               : pd.DataFrame
    categorical_cols : list of categorical column names
    top_n            : number of top values to include in frequency tables

    Returns
    -------
    dict  {column_name → analysis_dict}
    """
    if not categorical_cols:
        logger.warning("No categorical columns — skipping categorical univariate analysis.")
        return {}

    results: Dict[str, Any] = {}

    for col in categorical_cols:
        series = df[col].dropna()
        n_unique = int(series.nunique())
        value_counts = series.value_counts(normalize=False)
        freq_pct = series.value_counts(normalize=True)

        results[col] = {
            "n_unique": n_unique,
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 4),
            "top_values": value_counts.head(top_n).to_dict(),
            "top_values_pct": (freq_pct.head(top_n) * 100).round(4).to_dict(),
            "mode": str(value_counts.index[0]) if not value_counts.empty else None,
            "mode_freq_pct": round(float(freq_pct.iloc[0]) * 100, 4)
            if not freq_pct.empty
            else None,
            "is_binary": n_unique == 2,
            "entropy": float(stats.entropy(freq_pct.values)) if not freq_pct.empty else 0.0,
        }

    logger.info(
        "Categorical univariate analysis complete for %d columns.", len(results)
    )
    return results


# ===========================================================================
# HELPER 4: OUTLIER DETECTION (IQR METHOD)
# ===========================================================================

def detect_outliers_iqr(
    df: pd.DataFrame,
    numeric_cols: List[str],
    fence: float = IQR_OUTLIER_FENCE,
) -> Dict[str, Any]:
    """
    Identify outliers using the Tukey IQR fencing method.

    An observation is flagged as an outlier if it lies below
    ``Q1 - fence * IQR`` or above ``Q3 + fence * IQR``.

    Parameters
    ----------
    df           : pd.DataFrame
    numeric_cols : list of numeric column names
    fence        : IQR multiplier (default 1.5; use 3.0 for extreme outliers)

    Returns
    -------
    dict  {column_name → outlier_stats_dict}
    """
    if not numeric_cols:
        return {}

    results: Dict[str, Any] = {}
    subset = df[numeric_cols]

    q1 = subset.quantile(0.25)
    q3 = subset.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - fence * iqr
    upper = q3 + fence * iqr

    for col in numeric_cols:
        series = subset[col].dropna()
        mask_low = series < lower[col]
        mask_high = series > upper[col]
        n_outliers = int((mask_low | mask_high).sum())
        n_total = len(series)

        results[col] = {
            "q1": round(float(q1[col]), 6),
            "q3": round(float(q3[col]), 6),
            "iqr": round(float(iqr[col]), 6),
            "lower_fence": round(float(lower[col]), 6),
            "upper_fence": round(float(upper[col]), 6),
            "n_outliers_lower": int(mask_low.sum()),
            "n_outliers_upper": int(mask_high.sum()),
            "n_outliers_total": n_outliers,
            "outlier_pct": round(n_outliers / n_total * 100, 4) if n_total else 0.0,
            "has_outliers": n_outliers > 0,
        }

    logger.info("IQR outlier detection complete for %d columns.", len(results))
    return results


# ===========================================================================
# HELPER 5: CORRELATION ANALYSIS
# ===========================================================================

def compute_correlations(
    df: pd.DataFrame,
    numeric_cols: List[str],
    threshold: float = STRONG_CORR_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute the Pearson correlation matrix and identify strongly correlated pairs.

    Parameters
    ----------
    df           : pd.DataFrame
    numeric_cols : list of numeric column names
    threshold    : absolute correlation value to classify as "strong"

    Returns
    -------
    dict containing:
        - ``correlation_matrix`` : serialisable dict-of-dicts
        - ``strong_pairs``       : list of (col_a, col_b, corr_value) tuples
        - ``summary``            : high-level statistics
    """
    if len(numeric_cols) < 2:
        logger.warning("Fewer than 2 numeric columns — skipping correlation analysis.")
        return {"correlation_matrix": {}, "strong_pairs": [], "summary": {}}

    corr_matrix = df[numeric_cols].corr(method="pearson")
    corr_dict = corr_matrix.round(4).to_dict()

    # Extract strongly correlated pairs (upper triangle only, exclude diagonal)
    strong_pairs: List[Dict[str, Any]] = []
    cols = numeric_cols
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                strong_pairs.append(
                    {
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "correlation": round(float(val), 4),
                        "direction": "positive" if val > 0 else "negative",
                        "strength": "very_strong" if abs(val) >= 0.9 else "strong",
                    }
                )

    # Sort by absolute correlation descending
    strong_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    summary = {
        "n_numeric_features": len(numeric_cols),
        "n_strong_pairs": len(strong_pairs),
        "max_correlation": round(
            float(
                corr_matrix.where(
                    ~np.eye(len(numeric_cols), dtype=bool)
                ).abs().max().max()
            ),
            4,
        )
        if len(numeric_cols) > 1
        else None,
    }

    logger.info(
        "Correlation analysis complete — %d strong pairs found (threshold=%.2f).",
        len(strong_pairs),
        threshold,
    )
    return {
        "correlation_matrix": corr_dict,
        "strong_pairs": strong_pairs,
        "summary": summary,
    }


# ===========================================================================
# HELPER 6: FEATURE RELATIONSHIPS (BIVARIATE)
# ===========================================================================

def analyze_feature_relationships(
    df: pd.DataFrame,
    feature_types: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Compute bivariate statistics across feature-type combinations:

    - Numeric × Categorical  : group-level statistics (mean, median, std, count)
    - Categorical × Categorical : contingency table (value counts only for
      low-cardinality columns to avoid memory explosion)

    Parameters
    ----------
    df            : pd.DataFrame
    feature_types : output of :func:`detect_feature_types`

    Returns
    -------
    dict with keys ``"numeric_vs_categorical"`` and ``"categorical_vs_categorical"``
    """
    numeric_cols = feature_types.get("numeric", [])
    cat_cols = feature_types.get("categorical", [])

    results: Dict[str, Any] = {
        "numeric_vs_categorical": {},
        "categorical_vs_categorical": {},
    }

    # ── Numeric × Categorical ──────────────────────────────────────────────
    for cat in cat_cols:
        n_unique = df[cat].nunique()
        if n_unique > MAX_CONTINGENCY_CARDINALITY:
            continue  # skip very high cardinality groupby (expensive & unreadable)

        group_stats: Dict[str, Any] = {}
        for num in numeric_cols:
            try:
                grp = (
                    df.groupby(cat, observed=True)[num]
                    .agg(["mean", "median", "std", "count"])
                    .round(4)
                )
                group_stats[num] = grp.to_dict(orient="index")
            except Exception as exc:
                logger.debug("Skipping %s × %s: %s", cat, num, exc)

        if group_stats:
            results["numeric_vs_categorical"][cat] = group_stats

    # ── Categorical × Categorical ──────────────────────────────────────────
    low_card_cats = [
        c for c in cat_cols if df[c].nunique() <= MAX_CONTINGENCY_CARDINALITY
    ]

    for i in range(len(low_card_cats)):
        for j in range(i + 1, len(low_card_cats)):
            col_a, col_b = low_card_cats[i], low_card_cats[j]
            key = f"{col_a}_x_{col_b}"
            try:
                ct = pd.crosstab(df[col_a], df[col_b])
                results["categorical_vs_categorical"][key] = {
                    "contingency_table": ct.to_dict(),
                    "col_a": col_a,
                    "col_b": col_b,
                    "shape": list(ct.shape),
                }
            except Exception as exc:
                logger.debug("Contingency table skipped for %s: %s", key, exc)

    logger.info(
        "Feature relationship analysis complete — "
        "%d num×cat blocks, %d cat×cat blocks.",
        len(results["numeric_vs_categorical"]),
        len(results["categorical_vs_categorical"]),
    )
    return results


# ===========================================================================
# HELPER 7: MULTIVARIATE / MULTICOLLINEARITY ANALYSIS
# ===========================================================================

def multivariate_analysis(
    df: pd.DataFrame,
    numeric_cols: List[str],
    corr_data: Dict[str, Any],
    vif_threshold: float = VIF_THRESHOLD,
) -> Dict[str, Any]:
    """
    Detect multicollinearity and summarise correlation clusters.

    VIF Computation
    ---------------
    Variance Inflation Factor is computed using OLS regression of each
    numeric feature against all remaining numeric features.
    VIF > 5 indicates moderate multicollinearity.
    VIF > 10 indicates severe multicollinearity.

    Correlation Clusters
    --------------------
    Simple greedy clustering: two columns belong to the same cluster
    if |corr| >= threshold.

    Parameters
    ----------
    df            : pd.DataFrame
    numeric_cols  : list of numeric column names
    corr_data     : output of :func:`compute_correlations`
    vif_threshold : VIF above this value is flagged

    Returns
    -------
    dict with ``"vif"``, ``"multicollinear_features"``, ``"correlation_clusters"``
    """
    results: Dict[str, Any] = {
        "vif": {},
        "multicollinear_features": [],
        "correlation_clusters": [],
    }

    if len(numeric_cols) < 2:
        return results

    # ── VIF ────────────────────────────────────────────────────────────────
    clean_subset = df[numeric_cols].dropna()
    if len(clean_subset) < len(numeric_cols) + 1:
        logger.warning("Insufficient rows for VIF calculation — skipping.")
    else:
        from numpy.linalg import lstsq

        X = clean_subset.values
        vif_scores: Dict[str, float] = {}

        for idx, col in enumerate(numeric_cols):
            y = X[:, idx]
            X_rest = np.delete(X, idx, axis=1)
            # Add intercept
            X_rest_i = np.column_stack([np.ones(len(X_rest)), X_rest])
            try:
                coefs, _, _, _ = lstsq(X_rest_i, y, rcond=None)
                y_hat = X_rest_i @ coefs
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
                vif = 1 / (1 - r2) if r2 < 1.0 else np.inf
                vif_scores[col] = round(float(vif), 4)
            except Exception as exc:
                logger.debug("VIF skipped for %s: %s", col, exc)
                vif_scores[col] = np.nan

        results["vif"] = vif_scores
        results["multicollinear_features"] = [
            col
            for col, v in vif_scores.items()
            if not np.isnan(v) and v > vif_threshold
        ]

    # ── Correlation Clusters ───────────────────────────────────────────────
    strong_pairs = corr_data.get("strong_pairs", [])
    clusters: List[List[str]] = []

    for pair in strong_pairs:
        col_a, col_b = pair["col_a"], pair["col_b"]
        placed = False
        for cluster in clusters:
            if col_a in cluster or col_b in cluster:
                if col_a not in cluster:
                    cluster.append(col_a)
                if col_b not in cluster:
                    cluster.append(col_b)
                placed = True
                break
        if not placed:
            clusters.append([col_a, col_b])

    results["correlation_clusters"] = clusters

    logger.info(
        "Multivariate analysis complete — %d multicollinear features, %d clusters.",
        len(results["multicollinear_features"]),
        len(clusters),
    )
    return results


# ===========================================================================
# HELPER 8: DISTRIBUTION ANALYSIS
# ===========================================================================

def distribution_analysis(
    df: pd.DataFrame,
    numeric_cols: List[str],
    numeric_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify the distributional characteristics of every numeric feature.

    Outputs
    -------
    - ``skewed_features``          : |skewness| > SKEWNESS_THRESHOLD
    - ``heavy_tailed_features``    : excess kurtosis > KURTOSIS_THRESHOLD
    - ``log_transform_candidates`` : skewed & all-positive values
    - ``near_normal_features``     : low skewness and low kurtosis
    - ``normality_tests``          : Shapiro-Wilk p-value (sample ≤ 5 000 rows)

    Parameters
    ----------
    df            : pd.DataFrame
    numeric_cols  : list of numeric column names
    numeric_stats : output of :func:`numeric_univariate_analysis`

    Returns
    -------
    dict
    """
    results: Dict[str, Any] = {
        "skewed_features": [],
        "heavy_tailed_features": [],
        "log_transform_candidates": [],
        "near_normal_features": [],
        "normality_tests": {},
        "distribution_tags": {},
    }

    for col in numeric_cols:
        col_stats = numeric_stats.get(col, {})
        if "error" in col_stats:
            continue

        skewness = col_stats.get("skewness", 0.0)
        kurtosis = col_stats.get("kurtosis", 0.0)
        series = df[col].dropna()

        tags: List[str] = []

        if abs(skewness) > SKEWNESS_THRESHOLD:
            results["skewed_features"].append(
                {"column": col, "skewness": round(skewness, 4)}
            )
            tags.append("skewed")

        if kurtosis > KURTOSIS_THRESHOLD:
            results["heavy_tailed_features"].append(
                {"column": col, "kurtosis": round(kurtosis, 4)}
            )
            tags.append("heavy_tailed")

        if abs(skewness) > LOG_TRANSFORM_SKEW_THRESHOLD and (series > 0).all():
            results["log_transform_candidates"].append(col)
            tags.append("log_transform_candidate")

        if abs(skewness) <= 0.5 and abs(kurtosis) <= 1.0:
            results["near_normal_features"].append(col)
            tags.append("near_normal")

        # Normality test (Shapiro-Wilk, capped at 5 000 samples for speed)
        sample = series.sample(min(5_000, len(series)), random_state=42)
        try:
            stat, p_val = stats.shapiro(sample)
            results["normality_tests"][col] = {
                "shapiro_stat": round(float(stat), 6),
                "p_value": round(float(p_val), 6),
                "is_normal_95": p_val > 0.05,
            }
            if p_val > 0.05:
                tags.append("normal_distribution")
        except Exception:
            pass

        results["distribution_tags"][col] = tags

    logger.info("Distribution analysis complete.")
    return results


# ===========================================================================
# HELPER 9: CARDINALITY ANALYSIS
# ===========================================================================

def cardinality_analysis(
    df: pd.DataFrame,
    feature_types: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Detect high-cardinality and ID-like categorical features.

    A column is flagged as *ID-like* if its unique count equals the row count
    or its unique-to-total ratio exceeds 0.95 and dtype is object/string.

    Parameters
    ----------
    df            : pd.DataFrame
    feature_types : output of :func:`detect_feature_types`

    Returns
    -------
    dict
    """
    cat_cols = feature_types.get("categorical", [])
    n_rows = len(df)

    high_cardinality: List[Dict[str, Any]] = []
    id_like_columns: List[str] = []
    cardinality_profile: Dict[str, Any] = {}

    for col in cat_cols:
        n_unique = int(df[col].nunique())
        ratio = n_unique / n_rows if n_rows else 0.0

        cardinality_profile[col] = {
            "n_unique": n_unique,
            "unique_ratio": round(ratio, 4),
        }

        if n_unique >= HIGH_CARDINALITY_THRESHOLD:
            high_cardinality.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "unique_ratio": round(ratio, 4),
                }
            )

        if n_unique == n_rows or ratio >= 0.95:
            id_like_columns.append(col)

    logger.info(
        "Cardinality analysis complete — %d high-cardinality, %d ID-like.",
        len(high_cardinality),
        len(id_like_columns),
    )
    return {
        "high_cardinality_features": high_cardinality,
        "id_like_columns": id_like_columns,
        "cardinality_profile": cardinality_profile,
    }


# ===========================================================================
# HELPER 10: VISUALIZATION RECOMMENDATIONS
# ===========================================================================

# def generate_visualization_recommendations(
#     feature_types: Dict[str, List[str]],
#     numeric_stats: Dict[str, Any],
#     outlier_data: Dict[str, Any],
#     corr_data: Dict[str, Any],
#     distribution_data: Dict[str, Any],
#     cardinality_data: Dict[str, Any],
# ) -> List[Dict[str, Any]]:
#     """
#     Generate a prioritised list of visualization recommendations.

#     Each recommendation is a dict with:
#     - ``chart_type``  : e.g. "histogram", "boxplot", "scatter", ...
#     - ``columns``     : list of columns involved
#     - ``priority``    : "high" | "medium" | "low"
#     - ``reasoning``   : plain-English justification
#     - ``config_hints``: optional rendering hints for the downstream viz layer

#     Parameters
#     ----------
#     feature_types     : output of detect_feature_types
#     numeric_stats     : output of numeric_univariate_analysis
#     outlier_data      : output of detect_outliers_iqr
#     corr_data         : output of compute_correlations
#     distribution_data : output of distribution_analysis
#     cardinality_data  : output of cardinality_analysis

#     Returns
#     -------
#     list of recommendation dicts
#     """
#     recs: List[Dict[str, Any]] = []

#     numeric_cols = feature_types.get("numeric", [])
#     cat_cols = feature_types.get("categorical", [])
#     bool_cols = feature_types.get("boolean", [])
#     dt_cols = feature_types.get("datetime", [])

#     # ── Histograms for every numeric column ───────────────────────────────
#     for col in numeric_cols:
#         skew = numeric_stats.get(col, {}).get("skewness", 0.0)
#         tags = distribution_data.get("distribution_tags", {}).get(col, [])
#         recs.append(
#             {
#                 "chart_type": "histogram",
#                 "columns": [col],
#                 "priority": "high",
#                 "reasoning": (
#                     f"Numeric column '{col}' (skewness={skew:.2f}). "
#                     "Histograms reveal the shape, modality, and spread of distributions."
#                 ),
#                 "config_hints": {
#                     "bins": "auto",
#                     "kde_overlay": True,
#                     "log_scale_x": col in distribution_data.get("log_transform_candidates", []),
#                     "tags": tags,
#                 },
#             }
#         )

#     # ── Boxplots for columns with outliers ────────────────────────────────
#     for col, stats_dict in outlier_data.items():
#         if stats_dict.get("has_outliers"):
#             recs.append(
#                 {
#                     "chart_type": "boxplot",
#                     "columns": [col],
#                     "priority": "high",
#                     "reasoning": (
#                         f"'{col}' has {stats_dict['n_outliers_total']} outliers "
#                         f"({stats_dict['outlier_pct']}%). "
#                         "Boxplots visually summarise spread and flag extremes."
#                     ),
#                     "config_hints": {
#                         "show_fliers": True,
#                         "outlier_pct": stats_dict["outlier_pct"],
#                     },
#                 }
#             )

#     # ── Bar charts for low-cardinality categoricals ────────────────────────
#     high_card_cols = {
#         d["column"] for d in cardinality_data.get("high_cardinality_features", [])
#     }
#     for col in cat_cols:
#         if col not in high_card_cols:
#             recs.append(
#                 {
#                     "chart_type": "bar_chart",
#                     "columns": [col],
#                     "priority": "high",
#                     "reasoning": (
#                         f"Categorical '{col}' has low cardinality. "
#                         "Bar charts clearly show frequency distributions."
#                     ),
#                     "config_hints": {"sort_by": "frequency", "top_n": 20},
#                 }
#             )

#     # ── Scatter plots for strongly correlated pairs ────────────────────────
#     for pair in corr_data.get("strong_pairs", []):
#         recs.append(
#             {
#                 "chart_type": "scatter_plot",
#                 "columns": [pair["col_a"], pair["col_b"]],
#                 "priority": "high",
#                 "reasoning": (
#                     f"Strong {pair['direction']} correlation "
#                     f"(r={pair['correlation']}) between '{pair['col_a']}' and "
#                     f"'{pair['col_b']}'. Scatter plots reveal linear/non-linear trends."
#                 ),
#                 "config_hints": {
#                     "regression_line": True,
#                     "correlation": pair["correlation"],
#                 },
#             }
#         )

#     # ── Correlation heatmap (all numeric) ─────────────────────────────────
#     if len(numeric_cols) >= 3:
#         recs.append(
#             {
#                 "chart_type": "heatmap",
#                 "columns": numeric_cols,
#                 "priority": "high",
#                 "reasoning": (
#                     "Heatmap provides a single-view summary of all pairwise "
#                     "Pearson correlations across numeric features."
#                 ),
#                 "config_hints": {
#                     "annotate": len(numeric_cols) <= 15,
#                     "cmap": "coolwarm",
#                     "center": 0,
#                 },
#             }
#         )

#     # ── Pair plot for small numeric feature sets ───────────────────────────
#     if 2 <= len(numeric_cols) <= 8:
#         recs.append(
#             {
#                 "chart_type": "pair_plot",
#                 "columns": numeric_cols,
#                 "priority": "medium",
#                 "reasoning": (
#                     "Pair plot (scatterplot matrix) provides a comprehensive "
#                     "bivariate overview across all numeric features at once."
#                 ),
#                 "config_hints": {"diag": "kde", "hue": cat_cols[0] if cat_cols else None},
#             }
#         )

#     # ── Grouped boxplot: numeric × categorical ─────────────────────────────
#     low_card_cats = [c for c in cat_cols if c not in high_card_cols]
#     for cat in low_card_cats[:3]:  # limit to top 3 categoricals
#         for num in numeric_cols[:5]:  # limit to top 5 numerics
#             recs.append(
#                 {
#                     "chart_type": "grouped_boxplot",
#                     "columns": [num, cat],
#                     "priority": "medium",
#                     "reasoning": (
#                         f"Grouped boxplot of '{num}' by '{cat}' reveals "
#                         "distributional differences across groups."
#                     ),
#                     "config_hints": {"orient": "v", "show_points": True},
#                 }
#             )

#     # ── Boolean columns as bar/pie ─────────────────────────────────────────
#     for col in bool_cols:
#         recs.append(
#             {
#                 "chart_type": "bar_chart",
#                 "columns": [col],
#                 "priority": "low",
#                 "reasoning": f"Boolean '{col}' — bar chart shows True/False balance.",
#                 "config_hints": {"as_percentage": True},
#             }
#         )

#     # ── Time series plots ──────────────────────────────────────────────────
#     for col in dt_cols:
#         for num in numeric_cols[:3]:
#             recs.append(
#                 {
#                     "chart_type": "line_chart",
#                     "columns": [col, num],
#                     "priority": "high",
#                     "reasoning": (
#                         f"'{num}' plotted over datetime '{col}' to reveal "
#                         "temporal trends and seasonality."
#                     ),
#                     "config_hints": {"x": col, "y": num, "resample": "auto"},
#                 }
#             )

#     logger.info("Generated %d visualization recommendations.", len(recs))
#     return recs


# ===========================================================================
# DATASET SUMMARY
# ===========================================================================

def _build_dataset_summary(df: pd.DataFrame, feature_types: Dict[str, List[str]]) -> Dict[str, Any]:
    """Return a high-level summary of the dataset."""
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "total_cells": df.size,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 4),
        "n_numeric": len(feature_types.get("numeric", [])),
        "n_categorical": len(feature_types.get("categorical", [])),
        "n_datetime": len(feature_types.get("datetime", [])),
        "n_boolean": len(feature_types.get("boolean", [])),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


# ===========================================================================
# MAIN FUNCTION
# ===========================================================================

def perform_eda( state : AgentState ) -> AgentState:
    """
    Perform comprehensive Exploratory Data Analysis on a pre-cleaned DataFrame.

    This function orchestrates the full EDA pipeline and returns a structured
    dictionary ready for consumption by downstream visualization and reporting
    systems.

    The input DataFrame is assumed to be:
      - Already cleaned (no dirty strings, standardised nulls)
      - Type-corrected (correct dtypes assigned)
      - Deduplicated
      - Profiled for missing values

    Parameters
    ----------
    df : pd.DataFrame
        A cleaned, type-corrected, deduplicated DataFrame.

    Returns
    -------
    Dict[str, Any]
        A nested dictionary with the following top-level keys:

        ``dataset_summary``
            Row/column counts, memory usage, dtype overview.

        ``feature_types``
            Columns grouped into numeric, categorical, datetime, boolean.

        ``numeric_analysis``
            Per-column descriptive statistics including skewness and kurtosis.

        ``categorical_analysis``
            Frequency distributions, entropy, cardinality per categorical column.

        ``correlations``
            Pearson correlation matrix and strongly correlated pairs.

        ``outliers``
            IQR-based outlier counts and percentages per numeric column.

        ``distributions``
            Skewness flags, log-transform candidates, normality test results.

        ``cardinality``
            High-cardinality features and potential ID-like columns.

        ``multivariate``
            VIF scores, multicollinear features, correlation clusters.

        ``feature_relationships``
            Group statistics (numeric × categorical) and contingency tables
            (categorical × categorical).

        ``visualization_recommendations``
            Prioritised list of chart suggestions with column references and
            configuration hints.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If ``df`` is empty.

    Examples
    --------
    >>> import pandas as pd
    >>> from enterprise_eda import perform_enterprise_eda
    >>> df = pd.read_csv("sales_data_cleaned.csv")
    >>> eda_results = perform_enterprise_eda(df)
    >>> print(eda_results["dataset_summary"])
    >>> print(eda_results["visualization_recommendations"][0])
    """
    df=state["df"]
    
    logger.info("=" * 70)
    logger.info("Enterprise EDA pipeline starting...")
    logger.info("=" * 70)

    # ── 0. Validate input ─────────────────────────────────────────────────
    _validate_input(df)

    # ── 1. Feature type detection ─────────────────────────────────────────
    logger.info("[1/9] Detecting feature types...")
    feature_types = detect_feature_types(df)
    numeric_cols = feature_types["numeric"]
    cat_cols = feature_types["categorical"]

    # ── 2. Dataset summary ────────────────────────────────────────────────
    logger.info("[2/9] Building dataset summary...")
    dataset_summary = _build_dataset_summary(df, feature_types)

    # ── 3. Numeric univariate analysis ────────────────────────────────────
    logger.info("[3/9] Running numeric univariate analysis...")
    numeric_analysis = numeric_univariate_analysis(df, numeric_cols)

    # ── 4. Categorical univariate analysis ───────────────────────────────
    logger.info("[4/9] Running categorical univariate analysis...")
    categorical_analysis = categorical_univariate_analysis(df, cat_cols)

    # ── 5. Correlation analysis ──────────────────────────────────────────
    logger.info("[5/9] Computing correlations...")
    correlations = compute_correlations(df, numeric_cols)

    # ── 6. Outlier detection ─────────────────────────────────────────────
    logger.info("[6/9] Detecting outliers (IQR method)...")
    outliers = detect_outliers_iqr(df, numeric_cols)

    # ── 7. Distribution analysis ─────────────────────────────────────────
    logger.info("[7/9] Analysing distributions...")
    distributions = distribution_analysis(df, numeric_cols, numeric_analysis)

    # ── 8. Cardinality analysis ──────────────────────────────────────────
    logger.info("[8/9] Running cardinality analysis...")
    cardinality = cardinality_analysis(df, feature_types)

    # ── 8b. Multivariate analysis ─────────────────────────────────────────
    multivariate = multivariate_analysis(df, numeric_cols, correlations)

    # ── 8c. Feature relationships ─────────────────────────────────────────
    feature_relationships = analyze_feature_relationships(df, feature_types)

    # ── 9. Visualization recommendations ─────────────────────────────────
    # logger.info("[9/9] Generating visualization recommendations...")
    # viz_recs = generate_visualization_recommendations(
    #     feature_types=feature_types,
    #     numeric_stats=numeric_analysis,
    #     outlier_data=outliers,
    #     corr_data=correlations,
    #     distribution_data=distributions,
    #     cardinality_data=cardinality,
    # )

    # ── Assemble final output ─────────────────────────────────────────────
    eda_result: Dict[str, Any] = {
        "dataset_summary": dataset_summary,
        "feature_types": feature_types,
        "numeric_analysis": numeric_analysis,
        "categorical_analysis": categorical_analysis,
        "correlations": correlations,
        "outliers": outliers,
        "distributions": distributions,
        "cardinality": cardinality,
        "multivariate": multivariate,
        "feature_relationships": feature_relationships,
        # "visualization_recommendations": viz_recs,
    }

    logger.info(
        "EDA pipeline complete. %d sections, %d viz recommendations.",
        len(eda_result),
        # len(viz_recs),
    )
    return {'eda_result': eda_result}


