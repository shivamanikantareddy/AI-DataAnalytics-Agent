"""
eda_agent_summary.py
====================
Production-grade module that converts a full Exploratory Data Analysis (EDA)
result dictionary into a compact, LLM-friendly analytical summary for use by
a Data Analysis Agent.

This module does NOT compute EDA. It purely transforms an existing EDA result
(potentially gigabytes of nested statistics) into a token-efficient signal
summary (≈200–500 tokens) that an AI agent can reason over to decide which
analytical tools to invoke next.

Design Goals
------------
- Token efficiency  : Output is always small regardless of input size.
- Deterministic schema : Fixed keys every caller can rely on.
- Robustness        : Handles missing/partial EDA sections gracefully.
- Scalability       : Works from 5 to 500+ column datasets.
- Zero heavy deps   : Standard library only.

Typical Integration
-------------------
    eda_result = enterprise_eda_pipeline.run(dataframe)   # existing pipeline
    summary    = build_agent_summary(eda_result)
    agent.decide_next_tool(summary)

Author  : Senior Data Platform Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level logger
# Consumers can configure the root logger or this logger directly.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# Private helper functions
# ===========================================================================


def _extract_dataset_overview(eda_result: Dict[str, Any]) -> Dict[str, int]:
    """Extract high-level dataset dimensions from the EDA result.

    Reads ``eda_result["dataset_summary"]`` and returns the row and column
    counts.  Falls back to ``0`` if the section or individual keys are absent
    so that downstream consumers always receive a valid integer.

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.

    Returns:
        A dictionary with keys:
            - ``rows``    (int): Total row count.
            - ``columns`` (int): Total column count.
    """
    summary = eda_result.get("dataset_summary", {})

    if not summary:
        logger.warning(
            "Section 'dataset_summary' is missing from the EDA result. "
            "Dataset overview will default to zeros."
        )

    rows = int(summary.get("rows", 0))
    columns = int(summary.get("n_columns", 0))  
    
    logger.debug("Dataset overview extracted: rows=%d, columns=%d", rows, columns)
    return {"rows": rows, "columns": columns}


def _extract_feature_types(
    eda_result: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Extract column lists grouped by their inferred data type.

    Reads ``eda_result["feature_types"]`` and returns four mutually exclusive
    lists that an agent uses to understand what kinds of analysis are
    applicable (e.g. regression on numeric, NLP on high-cardinality
    categorical, etc.).

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.

    Returns:
        A dictionary with keys:
            - ``numeric``     (List[str]): Continuous / integer columns.
            - ``categorical`` (List[str]): String / factor columns.
            - ``datetime``    (List[str]): Temporal columns.
            - ``boolean``     (List[str]): True/False flag columns.
    """
    feature_types = eda_result.get("feature_types", {})

    if not feature_types:
        logger.warning(
            "Section 'feature_types' is missing from the EDA result. "
            "Feature type lists will be empty."
        )

    numeric: List[str] = list(feature_types.get("numeric", []))
    categorical: List[str] = list(feature_types.get("categorical", []))
    datetime: List[str] = list(feature_types.get("datetime", []))
    boolean: List[str] = list(feature_types.get("boolean", []))

    logger.debug(
        "Feature types extracted: numeric=%d, categorical=%d, "
        "datetime=%d, boolean=%d",
        len(numeric),
        len(categorical),
        len(datetime),
        len(boolean),
    )
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime,
        "boolean": boolean,
    }


def _extract_strong_correlations(
    eda_result: Dict[str, Any],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Extract the top-N strongest pairwise correlations.

    Full correlation matrices are too large to pass to an LLM agent directly.
    This helper distils the matrix down to only the most actionable pairs,
    sorted by absolute correlation strength descending.

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.
        top_n: Maximum number of correlation pairs to return. Defaults to 10.

    Returns:
        List of dicts, each with keys:
            - ``col_a``       (str):   First column name.
            - ``col_b``       (str):   Second column name.
            - ``correlation`` (float): Pearson / Spearman coefficient.

        Returns an empty list when the section is absent.
    """
    correlations = eda_result.get("correlations", {})
    strong_pairs: List[Any] = correlations.get("strong_pairs", [])

    if not correlations:
        logger.warning(
            "Section 'correlations' is missing from the EDA result. "
            "Strong correlations will be empty."
        )

    # Normalise each pair to a canonical dict shape and sort by |correlation|
    extracted: List[Dict[str, Any]] = []
    for pair in strong_pairs:
        if not isinstance(pair, dict):
            # Skip malformed entries defensively
            continue
        col_a = pair.get("col_a") or pair.get("feature_a") or pair.get("column_a", "")
        col_b = pair.get("col_b") or pair.get("feature_b") or pair.get("column_b", "")
        corr_value = pair.get("correlation") or pair.get("value") or pair.get("corr", 0.0)

        if col_a and col_b:
            extracted.append(
                {"col_a": col_a, "col_b": col_b, "correlation": float(corr_value)}
            )

    # Sort by absolute correlation strength, strongest first
    extracted.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    result = extracted[:top_n]

    logger.debug("Strong correlations extracted: %d pairs (capped at %d)", len(result), top_n)
    return result


def _extract_multicollinear_features(eda_result: Dict[str, Any]) -> List[str]:
    """Extract column names flagged as multicollinear.

    Multicollinear features should be removed or combined before regression
    modelling. The agent uses this list to decide whether to invoke a VIF
    (Variance Inflation Factor) or PCA tool.

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.

    Returns:
        List of column names (str) flagged as multicollinear.
    """
    multivariate = eda_result.get("multivariate", {})
    features: List[str] = list(multivariate.get("multicollinear_features", []))

    if not multivariate:
        logger.warning("Section 'multivariate' is missing from the EDA result.")

    logger.debug("Multicollinear features extracted: %d", len(features))
    return features


def _extract_skewed_features(eda_result: Dict[str, Any]) -> List[str]:
    """Extract column names with high skewness.

    Skewed features often require transformation (log, Box-Cox, etc.) before
    modelling. The agent uses this list to decide whether to invoke a
    normalisation or transformation tool.

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.

    Returns:
        List of column names (str) with high skewness.
    """
    distributions = eda_result.get("distributions", {})
    skewed_raw = distributions.get("skewed_features", [])

    # skewed_features may be a list of strings or a list of dicts {col, skew}
    skewed_cols: List[str] = []
    for item in skewed_raw:
        if isinstance(item, str):
            skewed_cols.append(item)
        elif isinstance(item, dict):
            col = item.get("column") or item.get("col") or item.get("feature", "")
            if col:
                skewed_cols.append(col)

    if not distributions:
        logger.warning("Section 'distributions' is missing from the EDA result.")

    logger.debug("Skewed features extracted: %d", len(skewed_cols))
    return skewed_cols


def _extract_log_transform_candidates(eda_result: Dict[str, Any]) -> List[str]:
    """Extract column names recommended for log transformation.

    These are a subset of skewed features where a log transform is specifically
    appropriate (e.g. strictly positive values with right skew).

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.

    Returns:
        List of column names (str) that are log-transform candidates.
    """
    distributions = eda_result.get("distributions", {})
    candidates_raw = distributions.get("log_transform_candidates", [])

    candidates: List[str] = []
    for item in candidates_raw:
        if isinstance(item, str):
            candidates.append(item)
        elif isinstance(item, dict):
            col = item.get("column") or item.get("col") or item.get("feature", "")
            if col:
                candidates.append(col)

    logger.debug("Log transform candidates extracted: %d", len(candidates))
    return candidates


def _extract_outlier_columns(eda_result: Dict[str, Any]) -> List[str]:
    """Extract column names where outliers were detected.

    Reads the ``outliers`` section and returns only columns explicitly flagged
    with ``has_outliers == True``.  The agent uses this to decide whether to
    invoke an outlier treatment or capping tool.

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.

    Returns:
        List of column names (str) containing detected outliers.
    """
    outliers_section = eda_result.get("outliers", {})

    if not outliers_section:
        logger.warning("Section 'outliers' is missing from the EDA result.")
        return []

    flagged: List[str] = []

    # Support two common shapes:
    # 1. Dict[col_name -> {"has_outliers": bool, ...}]
    # 2. List[{"column": str, "has_outliers": bool, ...}]
    if isinstance(outliers_section, dict):
        for col, meta in outliers_section.items():
            if isinstance(meta, dict) and meta.get("has_outliers", False):
                flagged.append(col)
    elif isinstance(outliers_section, list):
        for entry in outliers_section:
            if isinstance(entry, dict) and entry.get("has_outliers", False):
                col = entry.get("column") or entry.get("col", "")
                if col:
                    flagged.append(col)

    logger.debug("Outlier columns extracted: %d", len(flagged))
    return flagged


def _extract_high_cardinality_features(eda_result: Dict[str, Any]) -> List[str]:
    """Extract column names with high cardinality.

    High-cardinality categorical columns are poor candidates for one-hot
    encoding and may require target encoding, hashing, or embedding. The
    agent uses this signal to avoid recommending naive encoding strategies.

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.

    Returns:
        List of column names (str) with high cardinality.
    """
    cardinality = eda_result.get("cardinality", {})

    if not cardinality:
        logger.warning("Section 'cardinality' is missing from the EDA result.")

    raw: List[Any] = cardinality.get("high_cardinality_features", [])
    cols: List[str] = [item if isinstance(item, str) else item.get("column", "") for item in raw]
    cols = [c for c in cols if c]  # drop empty strings

    logger.debug("High cardinality features extracted: %d", len(cols))
    return cols


def _extract_id_like_columns(eda_result: Dict[str, Any]) -> List[str]:
    """Extract column names that appear to be ID / surrogate key columns.

    ID-like columns (e.g. monotonically increasing integers, UUIDs) should be
    excluded from feature engineering and modelling.  The agent uses this list
    to avoid recommending statistical analysis on non-informative identifiers.

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.

    Returns:
        List of column names (str) that look like IDs.
    """
    cardinality = eda_result.get("cardinality", {})
    raw: List[Any] = cardinality.get("id_like_columns", [])
    cols: List[str] = [item if isinstance(item, str) else item.get("column", "") for item in raw]
    cols = [c for c in cols if c]

    logger.debug("ID-like columns extracted: %d", len(cols))
    return cols


def _extract_analysis_signals(
    eda_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate all low-level analytical signals into a single dictionary.

    This is the central signal extraction step.  Each signal type calls a
    dedicated private extractor so that the logic stays modular and testable.

    Args:
        eda_result: Full EDA result dictionary produced by the EDA pipeline.

    Returns:
        A dictionary with keys:
            - ``strong_correlations``      : Top correlated feature pairs.
            - ``multicollinear_features``  : Features with high VIF / collinearity.
            - ``skewed_features``          : Columns with high skewness.
            - ``log_transform_candidates`` : Columns suited for log transform.
            - ``outlier_columns``          : Columns containing outliers.
            - ``high_cardinality_features``: Columns with very many unique values.
            - ``id_like_columns``          : Columns that appear to be identifiers.
    """
    strong_correlations = _extract_strong_correlations(eda_result)
    multicollinear = _extract_multicollinear_features(eda_result)
    skewed = _extract_skewed_features(eda_result)
    log_candidates = _extract_log_transform_candidates(eda_result)
    outlier_cols = _extract_outlier_columns(eda_result)
    high_card = _extract_high_cardinality_features(eda_result)
    id_like = _extract_id_like_columns(eda_result)

    logger.info(
        "Analysis signals extracted — correlations: %d, multicollinear: %d, "
        "skewed: %d, log_candidates: %d, outliers: %d, high_card: %d, id_like: %d",
        len(strong_correlations),
        len(multicollinear),
        len(skewed),
        len(log_candidates),
        len(outlier_cols),
        len(high_card),
        len(id_like),
    )

    return {
        "strong_correlations": strong_correlations,
        "multicollinear_features": multicollinear,
        "skewed_features": skewed,
        "log_transform_candidates": log_candidates,
        "outlier_columns": outlier_cols,
        "high_cardinality_features": high_card,
        "id_like_columns": id_like,
    }


def _detect_analysis_opportunities(
    feature_types: Dict[str, List[str]],
    signals: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Derive higher-level analytical opportunity signals for the agent.

    Rather than exposing raw statistics, this function translates feature
    metadata and signal flags into actionable analysis categories that an
    agent can map directly to tool calls (e.g. "invoke regression tool" or
    "invoke clustering tool").

    Rules
    -----
    - **Regression candidates**: numeric columns that appear in at least one
      strong correlation pair.  These columns have demonstrated linear
      relationships and are likely informative target / predictor variables.

    - **Segmentation candidates**: categorical columns that are NOT flagged
      as high-cardinality and NOT flagged as ID-like.  Low-to-medium
      cardinality categoricals partition the dataset into meaningful groups
      for clustering or stratified analysis.

    - **Time series candidates**: any datetime column detected in the dataset.
      These columns enable temporal analysis, trend detection, and forecasting.

    - **Classification candidates**: boolean columns.  Binary targets are the
      simplest and most direct signal that a classification model is applicable.

    Args:
        feature_types: Output of ``_extract_feature_types()``.
        signals:       Output of ``_extract_analysis_signals()``.

    Returns:
        A dictionary with keys:
            - ``regression_candidates``    (List[str])
            - ``segmentation_candidates``  (List[str])
            - ``time_series_candidates``   (List[str])
            - ``classification_candidates`` (List[str])
    """
    # ------------------------------------------------------------------
    # Regression candidates
    # Collect all column names that participate in a strong correlation.
    # These are the most analytically interesting numeric features.
    # ------------------------------------------------------------------
    corr_cols: set[str] = set()
    for pair in signals.get("strong_correlations", []):
        col_a: str = pair.get("col_a", "")
        col_b: str = pair.get("col_b", "")
        if col_a:
            corr_cols.add(col_a)
        if col_b:
            corr_cols.add(col_b)

    numeric_set = set(feature_types.get("numeric", []))
    regression_candidates: List[str] = sorted(corr_cols & numeric_set)

    # ------------------------------------------------------------------
    # Segmentation candidates
    # Low-to-medium cardinality categoricals that are not noisy IDs.
    # Agents should not segment on high-cardinality or identifier columns.
    # ------------------------------------------------------------------
    high_card_set = set(signals.get("high_cardinality_features", []))
    id_like_set = set(signals.get("id_like_columns", []))
    exclude_for_segmentation = high_card_set | id_like_set

    segmentation_candidates: List[str] = [
        col
        for col in feature_types.get("categorical", [])
        if col not in exclude_for_segmentation
    ]

    # ------------------------------------------------------------------
    # Time series candidates
    # Any datetime column can anchor a temporal analysis workflow.
    # ------------------------------------------------------------------
    time_series_candidates: List[str] = list(feature_types.get("datetime", []))

    # ------------------------------------------------------------------
    # Classification candidates
    # Boolean columns are binary targets; recommend classification tooling.
    # ------------------------------------------------------------------
    classification_candidates: List[str] = list(feature_types.get("boolean", []))

    logger.debug(
        "Analysis opportunities detected — regression: %d, segmentation: %d, "
        "time_series: %d, classification: %d",
        len(regression_candidates),
        len(segmentation_candidates),
        len(time_series_candidates),
        len(classification_candidates),
    )

    return {
        "regression_candidates": regression_candidates,
        "segmentation_candidates": segmentation_candidates,
        "time_series_candidates": time_series_candidates,
        "classification_candidates": classification_candidates,
    }


# ===========================================================================
# Public API
# ===========================================================================


def build_agent_summary(eda_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a full EDA result dictionary into a compact agent summary.

    This is the single public entry point of the module. It orchestrates all
    private extraction helpers and returns a stable, token-efficient summary
    that an AI Data Analysis Agent can use to decide which analytical tools to
    invoke next.

    The output schema is always identical regardless of which sections are
    present in ``eda_result``, ensuring that agents can safely access any key
    without defensive checks on their side.

    Args:
        eda_result: Large nested dictionary produced by the enterprise EDA
            pipeline.  Expected top-level keys include (but are not limited
            to): ``dataset_summary``, ``feature_types``, ``numeric_analysis``,
            ``categorical_analysis``, ``correlations``, ``outliers``,
            ``distributions``, ``cardinality``, ``multivariate``, and
            ``feature_relationships``.  Any subset of these keys is acceptable;
            missing sections are handled gracefully.

    Returns:
        A compact summary dictionary with the following stable schema::

            {
                "dataset_overview": {
                    "rows": int,
                    "columns": int
                },
                "feature_types": {
                    "numeric": List[str],
                    "categorical": List[str],
                    "datetime": List[str],
                    "boolean": List[str]
                },
                "analysis_signals": {
                    "strong_correlations": List[Dict],
                    "multicollinear_features": List[str],
                    "skewed_features": List[str],
                    "log_transform_candidates": List[str],
                    "outlier_columns": List[str],
                    "high_cardinality_features": List[str],
                    "id_like_columns": List[str]
                },
                "analysis_opportunities": {
                    "regression_candidates": List[str],
                    "segmentation_candidates": List[str],
                    "time_series_candidates": List[str],
                    "classification_candidates": List[str]
                }
            }

    Raises:
        TypeError: If ``eda_result`` is not a dictionary.

    Example::

        eda_result = enterprise_eda_pipeline.run(df)
        summary = build_agent_summary(eda_result)
        agent.decide_next_tool(summary)
    """
    if not isinstance(eda_result, dict):
        raise TypeError(
            f"eda_result must be a dict, got {type(eda_result).__name__!r}."
        )

    logger.info("Starting EDA agent summary generation.")

    # Step 1: Dataset-level metadata
    dataset_overview = _extract_dataset_overview(eda_result)

    # Step 2: Typed feature lists
    feature_types = _extract_feature_types(eda_result)
    total_features = sum(len(v) for v in feature_types.values())
    logger.info("Total features detected across all types: %d", total_features)

    # Step 3: Statistical signals (distilled from large EDA sections)
    analysis_signals = _extract_analysis_signals(eda_result)

    # Step 4: Higher-level opportunities derived from signals + feature types
    analysis_opportunities = _detect_analysis_opportunities(
        feature_types, analysis_signals
    )

    summary: Dict[str, Any] = {
        "dataset_overview": dataset_overview,
        "feature_types": feature_types,
        "analysis_signals": analysis_signals,
        "analysis_opportunities": analysis_opportunities,
    }

    logger.info("EDA agent summary generation complete.")
    return summary


# ===========================================================================
# Example usage / smoke test
# ===========================================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # -----------------------------------------------------------------------
    # Minimal synthetic EDA result that mirrors the real pipeline's schema.
    # In production this dict comes from the enterprise EDA pipeline.
    # -----------------------------------------------------------------------
    eda_result: Dict[str, Any] = {
        "dataset_summary": {
            "rows": 120_000,
            "columns": 47,
        },
        "feature_types": {
            "numeric": [
                "age", "income", "loan_amount", "credit_score",
                "debt_ratio", "num_accounts",
            ],
            "categorical": [
                "employment_status", "loan_purpose", "state",
                "product_category", "customer_segment",
            ],
            "datetime": ["account_open_date", "last_transaction_date"],
            "boolean": ["is_defaulted", "is_fraud"],
        },
        "correlations": {
            "strong_pairs": [
                {"col_a": "income",       "col_b": "loan_amount",   "correlation": 0.87},
                {"col_a": "credit_score", "col_b": "debt_ratio",    "correlation": -0.74},
                {"col_a": "income",       "col_b": "credit_score",  "correlation": 0.61},
                {"col_a": "age",          "col_b": "num_accounts",  "correlation": 0.55},
                {"col_a": "loan_amount",  "col_b": "debt_ratio",    "correlation": 0.49},
            ]
        },
        "multivariate": {
            "multicollinear_features": ["income", "loan_amount", "debt_ratio"],
        },
        "distributions": {
            "skewed_features": [
                "income", "loan_amount", "num_accounts",
            ],
            "log_transform_candidates": ["income", "loan_amount"],
        },
        "outliers": {
            "income":       {"has_outliers": True,  "method": "IQR"},
            "age":          {"has_outliers": False, "method": "IQR"},
            "loan_amount":  {"has_outliers": True,  "method": "IQR"},
            "credit_score": {"has_outliers": False, "method": "IQR"},
            "debt_ratio":   {"has_outliers": True,  "method": "Z-score"},
        },
        "cardinality": {
            "high_cardinality_features": ["state", "product_category"],
            "id_like_columns":           ["customer_id", "loan_id"],
        },
    }

    summary = build_agent_summary(eda_result)
    print("\n" + "=" * 70)
    print("EDA AGENT SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70)