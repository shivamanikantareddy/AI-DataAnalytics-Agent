"""
Data Cleaning Profiler
======================
Analyzes a pandas DataFrame and produces structured signals for an autonomous
data cleaning agent to decide which cleaning operations to apply.

NOT an EDA tool — all signals are action-oriented.
"""
from utils.state import AgentState 

import warnings
import numpy as np
import pandas as pd
from typing import Any

warnings.filterwarnings("ignore")



# ---------------------------------------------------------------------------
# 1. DATASET OVERVIEW
# ---------------------------------------------------------------------------

def _analyze_dataset_overview(df: pd.DataFrame) -> dict:
    """High-level signals about the shape and structure of the DataFrame."""
    row_count, col_count = df.shape

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Duplicate columns (identical values across all rows)
    seen: dict[str, str] = {}
    duplicate_columns: list[str] = []
    for col in df.columns:
        key = df[col].astype(str).tolist().__str__()
        if key in seen:
            duplicate_columns.append(col)
        else:
            seen[key] = col

    # Column name issues: spaces, hyphens, uppercase letters
    column_name_issues = [
        col for col in df.columns
        if " " in str(col) or "-" in str(col) or any(c.isupper() for c in str(col))
    ]

    # dtype map
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    return {
        "row_count": row_count,
        "column_count": col_count,
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_pct": round(duplicate_rows / row_count * 100, 4) if row_count else 0.0,
        "duplicate_columns": duplicate_columns,
        "column_name_issues": column_name_issues,
        "dtypes": dtypes,
    }


# ---------------------------------------------------------------------------
# 2. COMPLETENESS ANALYSIS
# ---------------------------------------------------------------------------

def _analyze_completeness(df: pd.DataFrame) -> dict:
    """Null / missing value signals per column and across the dataset."""
    row_count = len(df)

    null_counts: dict[str, int] = {}
    null_pcts: dict[str, float] = {}
    masked_null_columns: list[str] = []

    for col in df.columns:
        series = df[col]
        n_null = int(series.isna().sum())
        null_counts[col] = n_null
        null_pcts[col] = round(n_null / row_count * 100, 4) if row_count else 0.0

        # Whitespace-only strings act as masked nulls
        if series.dtype == object:
            ws_mask = series.dropna().astype(str).str.strip() == ""
            if ws_mask.any():
                masked_null_columns.append(col)

    rows_entirely_null = int((df.isna().all(axis=1)).sum())
    columns_above_50 = [c for c, p in null_pcts.items() if p > 50]
    columns_above_80 = [c for c, p in null_pcts.items() if p > 80]

    return {
        "null_counts": null_counts,
        "null_pcts": null_pcts,
        "masked_null_columns": masked_null_columns,
        "rows_entirely_null": rows_entirely_null,
        "columns_above_50pct_missing": columns_above_50,
        "columns_above_80pct_missing": columns_above_80,
    }


# ---------------------------------------------------------------------------
# 3. COLUMN TYPE INFERENCE
# ---------------------------------------------------------------------------

def _infer_column_type(series: pd.Series) -> str:
    """
    Returns one of: "numeric", "datetime", "categorical".
    Numeric / datetime checks take priority over categorical.
    """
    # Already a numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Already a datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if series.dtype == object:
        sample = series.dropna().head(100)
        if sample.empty:
            return "categorical"

        # Try numeric conversion first (faster than datetime)
        try:
            pd.to_numeric(sample)
            return "numeric"
        except (ValueError, TypeError):
            pass

        # Try datetime parsing
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return "datetime"
        except Exception:
            pass

    return "categorical"


# ---------------------------------------------------------------------------
# 4. NUMERIC COLUMN PROFILING
# ---------------------------------------------------------------------------

def _profile_numeric(series: pd.Series) -> dict:
    """Action-oriented numeric signals only — no EDA statistics."""
    clean = series.replace([np.inf, -np.inf], np.nan)
    non_null = clean.dropna()
    count_non_null = len(non_null)

    if count_non_null == 0:
        return {
            "count_non_null": 0,
            "min": None,
            "max": None,
            "skewness": None,
            "zero_ratio": None,
            "negative_ratio": None,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "inf_count": 0,
        }

    inf_count = int(np.isinf(series).sum())

    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (non_null < lower) | (non_null > upper)
    outlier_count = int(outlier_mask.sum())

    return {
        "count_non_null": count_non_null,
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "skewness": round(float(non_null.skew()), 6),
        "zero_ratio": round(float((non_null == 0).sum() / count_non_null), 6),
        "negative_ratio": round(float((non_null < 0).sum() / count_non_null), 6),
        "outlier_count": outlier_count,
        "outlier_ratio": round(outlier_count / count_non_null, 6),
        "inf_count": inf_count,
    }


# ---------------------------------------------------------------------------
# 5. CATEGORICAL COLUMN PROFILING
# ---------------------------------------------------------------------------

_BOOLEAN_TOKENS = {"true", "false", "yes", "no", "0", "1"}


def _profile_categorical(series: pd.Series) -> dict:
    """Action-oriented categorical signals."""
    non_null = series.dropna()
    count_non_null = len(non_null)
    unique_count = int(series.nunique(dropna=True))
    row_count = len(series)

    unique_ratio = round(unique_count / count_non_null, 6) if count_non_null else 0.0

    # Numeric-like ratio
    def _is_numeric(val):
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    numeric_like_ratio = (
        round(non_null.apply(_is_numeric).mean(), 6) if count_non_null else 0.0
    )

    # Boolean candidate
    lower_vals = set(non_null.astype(str).str.strip().str.lower().unique())
    boolean_candidate = lower_vals.issubset(_BOOLEAN_TOKENS) and len(lower_vals) >= 1

    return {
        "count_non_null": count_non_null,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "is_binary": unique_count == 2,
        "is_constant": unique_count <= 1,
        "is_identifier": unique_ratio > 0.98 and unique_count > 1,
        "numeric_like_ratio": numeric_like_ratio,
        "boolean_candidate": boolean_candidate,
    }


# ---------------------------------------------------------------------------
# 6. DATETIME COLUMN PROFILING
# ---------------------------------------------------------------------------

def _profile_datetime(series: pd.Series) -> dict:
    """Action-oriented datetime signals."""
    # Coerce to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    else:
        parsed = series

    non_null = parsed.dropna()
    count_non_null = len(non_null)
    now = pd.Timestamp.now()
    epoch_zero = pd.Timestamp("1970-01-01")

    future_dated_count = int((non_null > now).sum()) if count_non_null else 0
    epoch_zero_count = int((non_null == epoch_zero).sum()) if count_non_null else 0

    return {
        "count_non_null": count_non_null,
        "future_dated_count": future_dated_count,
        "epoch_zero_count": epoch_zero_count,
    }


# ---------------------------------------------------------------------------
# 7. DATA QUALITY FLAGS
# ---------------------------------------------------------------------------

def _detect_quality_flags(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Per-column quality flags that signal specific cleaning needs.
    Returns: { column_name: [flag1, flag2, ...] }
    """
    flags: dict[str, list[str]] = {col: [] for col in df.columns}

    for col in df.columns:
        series = df[col]
        col_flags = flags[col]

        # Mixed types in object columns
        if series.dtype == object:
            non_null = series.dropna()
            if not non_null.empty:
                type_set = set(type(v).__name__ for v in non_null)
                if len(type_set) > 1:
                    col_flags.append("mixed_types")

                # Whitespace values
                ws_count = (non_null.astype(str).str.strip() == "").sum()
                if ws_count > 0:
                    col_flags.append("whitespace_values")

        # Constant column (all non-null values are the same)
        unique_count = series.nunique(dropna=True)
        if unique_count <= 1:
            col_flags.append("constant_column")

        # Near-constant numeric (std ≈ 0 but not fully constant)
        if pd.api.types.is_numeric_dtype(series):
            clean = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) > 1 and unique_count > 1:
                std = float(clean.std())
                if std < 1e-9:
                    col_flags.append("near_constant_numeric")

    return flags


# ---------------------------------------------------------------------------
# 8. KEY ANALYSIS
# ---------------------------------------------------------------------------

def _analyze_keys(df: pd.DataFrame) -> dict:
    """Detect columns that could serve as primary keys."""
    row_count = len(df)
    candidate_primary_keys = [
        col for col in df.columns
        if df[col].nunique(dropna=False) == row_count and df[col].isna().sum() == 0
    ]
    return {"candidate_primary_keys": candidate_primary_keys}


# ---------------------------------------------------------------------------
# 9. UNIVARIATE ROUTER
# ---------------------------------------------------------------------------

def _build_univariate_profiles(df: pd.DataFrame) -> dict:
    """
    Route each column to the appropriate profiler and collect results.
    Returns: { column_name: { "inferred_type": ..., ...profile_fields } }
    """
    profiles: dict[str, dict] = {}

    for col in df.columns:
        series = df[col]
        inferred_type = _infer_column_type(series)

        if inferred_type == "numeric":
            profile = _profile_numeric(pd.to_numeric(series, errors="coerce"))
        elif inferred_type == "datetime":
            profile = _profile_datetime(series)
        else:
            profile = _profile_categorical(series)

        profiles[col] = {"inferred_type": inferred_type, **profile}

    return profiles


# ---------------------------------------------------------------------------
# 10. MASTER FUNCTION
# ---------------------------------------------------------------------------

def profile_dataframe(state: AgentState) -> AgentState:
    """
    Entry point for the data cleaning agent.

    Reads state["df"] (a pandas DataFrame) and returns:
        { "report": { ... structured cleaning signals ... } }
    """
    df: pd.DataFrame = state["df"]

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"state['df'] must be a pandas DataFrame, got {type(df)}")

    report = {
        "dataset_overview": _analyze_dataset_overview(df),
        "completeness_analysis": _analyze_completeness(df),
        "univariate_profiles": _build_univariate_profiles(df),
        "data_quality_flags": _detect_quality_flags(df),
        "key_analysis": _analyze_keys(df),
    }

    return {"report": report}


# import pandas as pd
# from ydata_profiling import ProfileReport


# def profile_dataframe(state: AgentState) -> AgentState:
#     # Dummy dataset
    

#     df = state['clean_df']

#     profile = ProfileReport(df)

#     report = profile.to_json()

#     print(report)