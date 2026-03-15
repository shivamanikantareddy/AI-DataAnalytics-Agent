import logging
import warnings
import re
from typing import Any, Dict, List, Optional, Tuple, Annotated

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
import featuretools as ft

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from utils.state import AgentState

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_cleaning_tools")


# ─────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────

def remove_first(lst: List[Any]) -> List[Any]:
    """Return a new list with the first element removed. Always capture the return value."""
    return lst[1:]


def _ensure_numeric(series: pd.Series) -> pd.Series:
    """
    If a series is object dtype but looks numeric, coerce it.
    Returns the series unchanged if already numeric.
    Raises ValueError with a clear message if coercion fails badly (>50% NaN).
    Used as a safety guard in any tool that requires numeric input.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[$€£¥₹%,\s]", "", regex=True)
        .str.replace(r"^\((.+)\)$", r"-\1", regex=True)
        .str.replace(r"[^\d.\-eE]", "", regex=True)
    )
    coerced = pd.to_numeric(cleaned, errors="coerce")
    null_rate = coerced.isna().mean()

    if null_rate > 0.5:
        raise ValueError(
            f"Column '{series.name}' could not be reliably converted to numeric "
            f"({null_rate:.1%} NaN after coercion). Run fix_dtypes first or choose a numeric column."
        )
    return coerced


# ─────────────────────────────────────────────
# INTERNAL HELPERS FOR fix_dtypes
# ─────────────────────────────────────────────

_DATE_HINTS = re.compile(
    r"(date|time|dt|year|month|day|timestamp|created|updated|born|"
    r"expir|modif|start|end|open|close)",
    re.IGNORECASE,
)

_BOOL_MAP = {
    "true": True, "false": False,
    "yes": True,  "no": False,
    "1": True,    "0": False,
    "y": True,    "n": False,
    "t": True,    "f": False,
}


def _to_numeric(series: pd.Series) -> pd.Series:
    """Strip common currency/percent/junk symbols then coerce to numeric."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[$€£¥₹%,\s]", "", regex=True)
        .str.replace(r"^\((.+)\)$", r"-\1", regex=True)
        .str.replace(r"[^\d.\-eE]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _to_datetime(series: pd.Series, fmt: str | None) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, format=fmt, errors="coerce", infer_datetime_format=True)


def _auto_convert(
    series: pd.Series,
    date_format: str | None,
    category_threshold: float,
) -> tuple[pd.Series, str]:
    """
    Try conversions in priority order:
      numeric -> datetime (name hint) -> datetime (sample probe) -> category -> unchanged.
    Null tolerance is 10% to handle real-world dirty CSVs.
    """
    null_before = series.isna().sum()
    max_new_nulls = max(1, int(len(series) * 0.10))

    numeric = _to_numeric(series)
    null_after = numeric.isna().sum()
    if (null_after - null_before) <= max_new_nulls and not numeric.isna().all():
        return numeric, "numeric"

    name_looks_like_date = bool(_DATE_HINTS.search(series.name or ""))
    date_conversion = _to_datetime(series, date_format)
    date_null_delta = date_conversion.isna().sum() - null_before

    if name_looks_like_date and date_null_delta <= max_new_nulls:
        return date_conversion, "datetime (name hint)"

    if not name_looks_like_date:
        sample = series.dropna().head(200)
        sample_converted = _to_datetime(sample, date_format)
        parse_rate = sample_converted.notna().mean()
        if parse_rate >= 0.80 and date_null_delta <= max_new_nulls:
            return date_conversion, "datetime (pattern probe)"

    n_unique = series.nunique(dropna=True)
    cardinality_ratio = n_unique / max(len(series), 1)
    if cardinality_ratio <= category_threshold:
        return series.astype("category"), "category (low cardinality)"

    return series, "unchanged"


def _downcast(series: pd.Series) -> pd.Series:
    """Downcast int64/float64 to the smallest lossless sub-type."""
    if pd.api.types.is_integer_dtype(series):
        return pd.to_numeric(series, downcast="integer")
    if pd.api.types.is_float_dtype(series):
        as32 = series.astype(np.float32)
        if np.allclose(series.fillna(0), as32.fillna(0), equal_nan=True, rtol=1e-5):
            return as32
    return series


def _try_bool(series: pd.Series) -> pd.Series | None:
    """Return a boolean Series if every non-null value maps to True/False."""
    lowered = series.dropna().astype(str).str.strip().str.lower()
    if lowered.empty:
        return None
    if set(lowered.unique()).issubset(_BOOL_MAP):
        return series.map(lambda v: _BOOL_MAP.get(str(v).strip().lower(), pd.NA))
    return None


# ─────────────────────────────────────────────
# 2. MISSING VALUE HANDLING TOOL
# ─────────────────────────────────────────────

VALID_STRATEGIES = {
    "mean", "median", "mode", "ffill", "bfill",
    "drop_rows", "drop_cols", "constant"
}


@tool
def handle_missing_values(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    strategy: str = "mean",
    columns: Optional[List[str]] = None,
    fill_value: Optional[Any] = None,
    drop_threshold: float = 0.5,
) -> Command:
    """
    Detect and handle missing values in a DataFrame.

    Parameters
    ----------
    strategy : str
        One of 'mean', 'median', 'mode', 'ffill', 'bfill',
        'drop_rows', 'drop_cols', 'constant'.
    columns : list of str, optional
        Columns to apply strategy to. Defaults to all columns.
    fill_value : any, optional
        Used when strategy='constant'. Required for that strategy.
    drop_threshold : float
        For 'drop_cols', fraction of missing values above which a column is dropped.

    Returns
    -------
    Command updating clean_df and cleaning_summary in state.
    """
    log = logging.getLogger("handle_missing_values")
    log.info("Starting | strategy=%s | columns=%s", strategy, columns)

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid options: {sorted(VALID_STRATEGIES)}"
        )
    if strategy == "constant" and fill_value is None:
        raise ValueError("fill_value must be provided when strategy='constant'.")

    df = state["clean_df"]
    result_df = df.copy()
    target_cols = columns if columns else df.columns.tolist()

    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    log.info("Target columns (%d): %s", len(target_cols), target_cols)
    imputation_log: Dict[str, Any] = {}

    if strategy == "mean":
        num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(result_df[c])]
        for col in num_cols:
            try:
                result_df[col] = _ensure_numeric(result_df[col])
            except ValueError as e:
                log.warning("Skipping column '%s': %s", col, e)
                continue
            missing = int(result_df[col].isnull().sum())
            if missing > 0:
                val = result_df[col].mean()
                result_df[col] = result_df[col].fillna(val)
                imputation_log[col] = {"strategy": "mean", "fill_value": float(val), "imputed_count": missing}
                log.info("Imputed '%s' with mean=%.4f (%d values)", col, val, missing)

    elif strategy == "median":
        num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(result_df[c])]
        for col in num_cols:
            try:
                result_df[col] = _ensure_numeric(result_df[col])
            except ValueError as e:
                log.warning("Skipping column '%s': %s", col, e)
                continue
            missing = int(result_df[col].isnull().sum())
            if missing > 0:
                val = result_df[col].median()
                result_df[col] = result_df[col].fillna(val)
                imputation_log[col] = {"strategy": "median", "fill_value": float(val), "imputed_count": missing}
                log.info("Imputed '%s' with median=%.4f (%d values)", col, val, missing)

    elif strategy == "mode":
        for col in target_cols:
            if pd.api.types.is_numeric_dtype(result_df[col]):
                unique_ratio = result_df[col].nunique() / max(len(result_df[col].dropna()), 1)
                if unique_ratio > 0.3:
                    log.warning(
                        "Column '%s' appears continuous. Mode imputation may be inappropriate. "
                        "Consider mean/median instead.", col
                    )
            missing = int(result_df[col].isnull().sum())
            if missing > 0:
                mode_vals = result_df[col].mode()
                if len(mode_vals) > 0:
                    val = mode_vals[0]
                    result_df[col] = result_df[col].fillna(val)
                    imputation_log[col] = {"strategy": "mode", "fill_value": str(val), "imputed_count": missing}
                    log.info("Imputed '%s' with mode='%s' (%d values)", col, val, missing)

    elif strategy == "ffill":
        for col in target_cols:
            before = int(result_df[col].isnull().sum())
            result_df[col] = result_df[col].ffill()
            after = int(result_df[col].isnull().sum())
            imputation_log[col] = {"strategy": "ffill", "imputed_count": before - after, "still_missing": after}
            log.info("ffill '%s': imputed=%d, still_missing=%d", col, before - after, after)

    elif strategy == "bfill":
        for col in target_cols:
            before = int(result_df[col].isnull().sum())
            result_df[col] = result_df[col].bfill()
            after = int(result_df[col].isnull().sum())
            imputation_log[col] = {"strategy": "bfill", "imputed_count": before - after, "still_missing": after}
            log.info("bfill '%s': imputed=%d, still_missing=%d", col, before - after, after)

    elif strategy == "constant":
        for col in target_cols:
            missing = int(result_df[col].isnull().sum())
            result_df[col] = result_df[col].fillna(fill_value)
            imputation_log[col] = {"strategy": "constant", "fill_value": str(fill_value), "imputed_count": missing}
            log.info("Filled '%s' with constant='%s' (%d values)", col, fill_value, missing)

    elif strategy == "drop_rows":
        before = len(result_df)
        result_df = result_df.dropna(subset=target_cols)
        dropped = before - len(result_df)
        imputation_log["dropped_rows"] = dropped
        log.info("Dropped %d rows with nulls in %s", dropped, target_cols)

    elif strategy == "drop_cols":
        dropped = []
        for col in target_cols:
            if result_df[col].isnull().mean() > drop_threshold:
                result_df = result_df.drop(columns=[col])
                dropped.append(col)
        imputation_log["dropped_columns"] = dropped
        log.info("Dropped columns: %s", dropped)

    remaining = int(result_df.isnull().sum().sum())
    missing_summary = {
        col: {
            "missing_count": int(result_df[col].isnull().sum()),
            "missing_pct": round(result_df[col].isnull().mean() * 100, 2),
        }
        for col in result_df.columns
        if result_df[col].isnull().sum() > 0
    }

    log.info("Completed | remaining_missing=%d", remaining)

    return Command(update={
        "clean_df": result_df,
        "cleaning_summary": {
            "handle_missing_values": {
                "strategy": strategy,
                "imputation_log": imputation_log,
                "remaining_missing": remaining,
                "missing_summary": missing_summary,
            }
        },
        "tool_priority_list_1": remove_first(state["tool_priority_list_1"]),
        "messages": [ToolMessage(
            content="handle_missing_values completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ─────────────────────────────────────────────
# 3. DUPLICATE DETECTION TOOL
# ─────────────────────────────────────────────

@tool
def detect_and_remove_duplicates(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    subset: Optional[List[str]] = None,
    keep: str = "first",
    remove: bool = True,
) -> Command:
    """
    Detect and optionally remove duplicate rows.

    Parameters
    ----------
    subset : list of str, optional
        Columns to consider for duplicate detection.
    keep : str
        'first', 'last', or False (drop all duplicates).
    remove : bool
        Whether to remove duplicates from the returned DataFrame.

    Returns
    -------
    Command updating clean_df and cleaning_summary in state.
    """
    log = logging.getLogger("detect_and_remove_duplicates")
    log.info("Starting | subset=%s | keep=%s | remove=%s", subset, keep, remove)

    df = state["clean_df"]
    dup_mask = df.duplicated(subset=subset, keep=False)
    n_duplicates = int(df.duplicated(subset=subset, keep=keep).sum())
    log.info("Detected %d duplicate rows", n_duplicates)

    duplicate_rows = df[dup_mask].copy()
    result_df = (
        df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        if remove else df.copy()
    )

    stats = {
        "original_row_count": len(df),
        "duplicate_row_count": n_duplicates,
        "rows_after_dedup": len(result_df),
        "rows_removed": len(df) - len(result_df),
        "duplicate_percentage": round(n_duplicates / max(len(df), 1) * 100, 2),
    }
    log.info(
        "Completed | rows_removed=%d | rows_remaining=%d | duplicate_pct=%.2f%%",
        stats["rows_removed"], stats["rows_after_dedup"], stats["duplicate_percentage"]
    )

    return Command(update={
        "clean_df": result_df,
        "cleaning_summary": {
            "detect_and_remove_duplicates": {
                "statistics": stats,
                "duplicate_rows_sample": duplicate_rows.head(20).to_dict(orient="records"),
            }
        },
        "tool_priority_list_1": remove_first(state["tool_priority_list_1"]),
        "messages": [ToolMessage(
            content="detect_and_remove_duplicates completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ─────────────────────────────────────────────
# 4. OUTLIER DETECTION TOOL
# ─────────────────────────────────────────────

@tool
def detect_outliers(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    method: str = "iqr",
    columns: Optional[List[str]] = None,
    action: str = "flag",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.05,
) -> Command:
    """
    Detect outliers using IQR, Z-score, or Isolation Forest.

    Parameters
    ----------
    method : str
        'iqr', 'zscore', or 'isolation_forest'.
    columns : list of str, optional
        Numeric columns to check. Defaults to all numeric columns.
    action : str
        'flag' (add boolean column), 'remove' (drop outlier rows), or 'cap' (winsorize).
    z_threshold : float
        Threshold for Z-score method.
    iqr_multiplier : float
        Multiplier for IQR bounds.
    contamination : float
        Isolation Forest contamination parameter.

    Returns
    -------
    Command updating clean_df and cleaning_summary in state.
    """
    log = logging.getLogger("detect_outliers")
    log.info("Starting | method=%s | action=%s | columns=%s", method, action, columns)

    df = state["clean_df"]
    result_df = df.copy()
    available_numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    if columns:
        num_cols = [col for col in columns if col in available_numeric]
        skipped = [col for col in columns if col not in available_numeric]
        for col in list(skipped):
            try:
                result_df[col] = _ensure_numeric(result_df[col])
                num_cols.append(col)
                skipped.remove(col)
                log.info("Coerced column '%s' to numeric", col)
            except ValueError as e:
                log.warning("Cannot coerce '%s': %s", col, e)
        if skipped:
            log.warning("Skipping non-numeric columns: %s", skipped)
    else:
        num_cols = available_numeric

    clean_num_cols = []
    for col in num_cols:
        try:
            result_df[col] = _ensure_numeric(result_df[col])
            clean_num_cols.append(col)
        except ValueError as e:
            log.warning("Skipping '%s': %s", col, e)
    num_cols = clean_num_cols
    log.info("Processing %d numeric columns", len(num_cols))

    outlier_counts: Dict[str, int] = {}
    outlier_indices: set = set()

    if method == "iqr":
        for col in num_cols:
            q1 = result_df[col].quantile(0.25)
            q3 = result_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask = (result_df[col] < lower) | (result_df[col] > upper)
            outlier_counts[col] = int(mask.sum())
            outlier_indices.update(result_df.index[mask].tolist())
            log.info("IQR '%s' | bounds=[%.4f, %.4f] | outliers=%d", col, lower, upper, outlier_counts[col])
            if action == "flag":
                result_df[f"{col}_outlier"] = mask.astype(int)
            elif action == "cap":
                result_df[col] = result_df[col].clip(lower=lower, upper=upper)

    elif method == "zscore":
        for col in num_cols:
            mean = result_df[col].mean()
            std = result_df[col].std()
            if std == 0:
                log.warning("Skipping '%s': std=0", col)
                continue
            z_scores = (result_df[col] - mean) / std
            mask = z_scores.abs() > z_threshold
            outlier_counts[col] = int(mask.sum())
            outlier_indices.update(result_df.index[mask].tolist())
            log.info("ZScore '%s' | threshold=%.2f | outliers=%d", col, z_threshold, outlier_counts[col])
            if action == "flag":
                result_df[f"{col}_outlier"] = mask.astype(int)
            elif action == "cap":
                lower = mean - z_threshold * std
                upper = mean + z_threshold * std
                result_df[col] = result_df[col].clip(lower=lower, upper=upper)

    elif method == "isolation_forest":
        valid_df = result_df[num_cols].dropna()
        if not valid_df.empty:
            clf = IsolationForest(contamination=contamination, random_state=42)
            preds = clf.fit_predict(valid_df)
            mask = pd.Series(preds == -1, index=valid_df.index)
            for col in num_cols:
                outlier_counts[col] = int(mask.sum())
            outlier_indices.update(valid_df.index[mask].tolist())
            log.info("IsolationForest | contamination=%.2f | total_outliers=%d", contamination, len(outlier_indices))
            if action == "flag":
                result_df["isolation_forest_outlier"] = 0
                result_df.loc[list(outlier_indices), "isolation_forest_outlier"] = 1

    if action == "remove" and outlier_indices:
        before = len(result_df)
        result_df = result_df.drop(index=list(outlier_indices)).reset_index(drop=True)
        log.info("Removed %d outlier rows", before - len(result_df))

    log.info("Completed | total_outlier_rows=%d", len(outlier_indices))

    return Command(update={
        "clean_df": result_df,
        "cleaning_summary": {
            "detect_outliers": {
                "method": method,
                "action": action,
                "outlier_counts_per_column": outlier_counts,
                "total_outlier_rows": len(outlier_indices),
            }
        },
        "tool_priority_list_1": remove_first(state["tool_priority_list_1"]),
        "messages": [ToolMessage(
            content="detect_outliers completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ─────────────────────────────────────────────
# 5. DATA TYPE CORRECTION TOOL
# ─────────────────────────────────────────────

@tool
def fix_dtypes(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    *,
    numeric_cols: list[str] | None = None,
    date_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    auto_detect: bool = True,
    date_format: str | None = None,
    category_threshold: float = 0.10,
    downcast_numerics: bool = True,
    inplace: bool = False,
) -> Command:
    """
    Convert and correct column data types like an expert data analyst.

    Parameters
    ----------
    numeric_cols : list of str, optional
        Columns to force-convert to numeric (coerce errors to NaN).
    date_cols : list of str, optional
        Columns to force-convert to datetime.
    categorical_cols : list of str, optional
        Columns to cast to pandas Categorical.
    auto_detect : bool, default True
        Scan every object/string column and attempt smart conversion.
    date_format : str, optional
        strftime format string passed to pd.to_datetime.
    category_threshold : float, default 0.10
        Auto-detect casts to Categorical when unique-value ratio <= this value.
    downcast_numerics : bool, default True
        Shrink int64/float64 to the smallest lossless sub-type.
    inplace : bool, default False
        If True, mutate df directly; otherwise work on a copy.

    Returns
    -------
    Command updating clean_df and cleaning_summary in state.
    """
    log = logging.getLogger("fix_dtypes")
    log.info(
        "Starting | auto_detect=%s | numeric_cols=%s | date_cols=%s | categorical_cols=%s",
        auto_detect, numeric_cols, date_cols, categorical_cols
    )

    df = state["clean_df"]
    out = df if inplace else df.copy()
    summary: dict[str, dict] = {}

    def _record(col: str, before: str, after: str, action: str) -> None:
        log.info("Converted '%s': %s -> %s (%s)", col, before, after, action)
        summary[col] = {"before": before, "after": after, "action": action}

    # ── 1. Explicit numeric columns ──────────────────────────────────────────
    for col in (numeric_cols or []):
        if col not in out.columns:
            log.warning("SKIP '%s' – column not found", col)
            continue
        before = str(out[col].dtype)
        out[col] = _to_numeric(out[col])
        after = str(out[col].dtype)
        if before != after:
            _record(col, before, after, "forced numeric")

    # ── 2. Explicit date columns ─────────────────────────────────────────────
    for col in (date_cols or []):
        if col not in out.columns:
            log.warning("SKIP '%s' – column not found", col)
            continue
        before = str(out[col].dtype)
        out[col] = _to_datetime(out[col], date_format)
        after = str(out[col].dtype)
        if before != after:
            _record(col, before, after, "forced datetime")

    # ── 3. Explicit categorical columns ─────────────────────────────────────
    for col in (categorical_cols or []):
        if col not in out.columns:
            log.warning("SKIP '%s' – column not found", col)
            continue
        before = str(out[col].dtype)
        out[col] = out[col].astype("category")
        _record(col, before, "category", "forced categorical")

    # ── 4. Auto-detection pass ───────────────────────────────────────────────
    if auto_detect:
        already_handled = set(
            (numeric_cols or []) + (date_cols or []) + (categorical_cols or [])
        )
        candidate_cols = [
            c for c in out.columns
            if c not in already_handled and out[c].dtype == object
        ]
        log.info("Auto-detect candidates (%d): %s", len(candidate_cols), candidate_cols)

        for col in candidate_cols:
            series = out[col]
            before = str(series.dtype)
            converted, action = _auto_convert(series, date_format, category_threshold)
            after = str(converted.dtype)
            if before != after:
                out[col] = converted
                _record(col, before, after, f"auto: {action}")

    # ── 5. Downcast numeric columns ──────────────────────────────────────────
    if downcast_numerics:
        for col in out.select_dtypes(include=["int64", "float64"]).columns:
            before = str(out[col].dtype)
            out[col] = _downcast(out[col])
            after = str(out[col].dtype)
            if before != after:
                _record(col, before, after, "downcast")

    # ── 6. Fix boolean-like object columns ───────────────────────────────────
    for col in out.select_dtypes(include="object").columns:
        if col in summary:
            continue
        converted = _try_bool(out[col])
        if converted is not None:
            before = str(out[col].dtype)
            out[col] = converted
            _record(col, before, "bool", "auto: boolean")

    # ── 7. Report remaining unconverted object columns ───────────────────────
    still_object = out.select_dtypes(include="object").columns.tolist()
    if still_object:
        log.warning("Still object dtype after all conversions: %s", still_object)
    else:
        log.info("All object columns successfully converted")

    if not summary:
        log.info("No dtype changes were necessary")

    log.info("Completed | %d columns converted", len(summary))

    return Command(update={
        "clean_df": out,
        "cleaning_summary": {
            "fix_dtypes": {
                "conversion_summary": summary,
                "still_object_cols": still_object,
                "total_converted": len(summary),
            }
        },
        "tool_priority_list_1": remove_first(state["tool_priority_list_1"]),
        "messages": [ToolMessage(
            content="fix_dtypes completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ─────────────────────────────────────────────
# 6. DATA STANDARDIZATION TOOL
# ─────────────────────────────────────────────

@tool
def standardize_data(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    text_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    fuzzy_map: Optional[Dict[str, List[str]]] = None,
    fuzzy_threshold: int = 85,
) -> Command:
    """
    Standardize text, categorical, and fuzzy-matched values.

    Parameters
    ----------
    text_cols : list of str, optional
        Columns to normalize (lowercase, strip whitespace).
    categorical_cols : list of str, optional
        Columns to standardize by mapping to canonical values (title-case).
    fuzzy_map : dict, optional
        {column: [canonical_value_1, canonical_value_2, ...]}.
        Values in the column are fuzzy-matched to the canonical list.
    fuzzy_threshold : int
        Minimum RapidFuzz score (0-100) for a match to be applied.

    Returns
    -------
    Command updating clean_df and cleaning_summary in state.
    """
    log = logging.getLogger("standardize_data")
    log.info(
        "Starting | text_cols=%s | categorical_cols=%s | fuzzy_map_keys=%s",
        text_cols, categorical_cols, list(fuzzy_map.keys()) if fuzzy_map else None
    )

    df = state["clean_df"]
    result_df = df.copy()
    change_log: Dict[str, Any] = {}

    # Text normalization
    for col in (text_cols or []):
        if col not in result_df.columns:
            log.warning("Text col '%s' not found, skipping", col)
            continue
        before = result_df[col].copy()
        result_df[col] = (
            result_df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        changed = int((result_df[col] != before.astype(str)).sum())
        change_log[col] = {"operation": "text_normalize", "cells_changed": changed}
        log.info("Text normalized '%s': %d cells changed", col, changed)

    # Categorical standardization
    for col in (categorical_cols or []):
        if col not in result_df.columns:
            log.warning("Categorical col '%s' not found, skipping", col)
            continue
        before = result_df[col].copy()
        result_df[col] = result_df[col].astype(str).str.strip().str.title()
        changed = int((result_df[col] != before.astype(str)).sum())
        change_log[col] = {"operation": "categorical_standardize", "cells_changed": changed}
        log.info("Categorical standardized '%s': %d cells changed", col, changed)

    # Fuzzy matching
    if fuzzy_map:
        for col, canonical_values in fuzzy_map.items():
            if col not in result_df.columns:
                log.warning("Fuzzy col '%s' not found, skipping", col)
                continue
            replacements: Dict[str, str] = {}
            for val in result_df[col].dropna().unique():
                match = process.extractOne(
                    str(val), canonical_values, scorer=fuzz.token_sort_ratio
                )
                if match and match[1] >= fuzzy_threshold:
                    if str(val) != match[0]:
                        replacements[str(val)] = match[0]
            result_df[col] = result_df[col].astype(str).replace(replacements)
            change_log[f"{col}_fuzzy"] = {
                "operation": "fuzzy_match",
                "replacements": replacements,
                "cells_changed": len(replacements),
            }
            log.info("Fuzzy matched '%s': %d replacements", col, len(replacements))

    log.info("Completed | %d columns processed", len(change_log))

    return Command(update={
        "clean_df": result_df,
        "cleaning_summary": {
            "standardize_data": {
                "change_log": change_log,
                "total_columns_processed": len(change_log),
            }
        },
        "tool_priority_list_1": remove_first(state["tool_priority_list_1"]),
        "messages": [ToolMessage(
            content="standardize_data completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ─────────────────────────────────────────────
# INTERNAL HELPER FOR FEATURETOOLS DFS
# ─────────────────────────────────────────────

def _run_deep_feature_synthesis(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    entity_id_col: Optional[str],
    agg_primitives: List[str],
    trans_primitives: List[str],
    max_depth: int,
) -> Tuple[pd.DataFrame, List[Any], int]:
    """Internal helper that runs Featuretools Deep Feature Synthesis on a single-entity dataset."""
    working_df = df[columns].copy() if columns else df.copy()

    if entity_id_col and entity_id_col in working_df.columns:
        working_df = working_df.set_index(entity_id_col)
    elif not isinstance(working_df.index, pd.RangeIndex):
        working_df = working_df.reset_index(drop=True)

    if working_df.index.name is None:
        working_df.index.name = "id"

    for col in working_df.select_dtypes(include=["object"]).columns:
        working_df[col] = working_df[col].astype("category")

    es = ft.EntitySet(id="dataset")
    es = es.add_dataframe(
        dataframe_name="main",
        dataframe=working_df,
        index=working_df.index.name,
        logical_types={
            col: ft.variable_types.Categorical
            for col in working_df.select_dtypes(include=["category"]).columns
        },
    )

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="main",
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        max_depth=max_depth,
        verbose=False,
        n_jobs=1,
    )

    feature_matrix = feature_matrix.reset_index(drop=True)

    if columns:
        excluded_cols = [c for c in df.columns if c not in columns]
        if excluded_cols:
            feature_matrix = pd.concat(
                [df[excluded_cols].reset_index(drop=True), feature_matrix], axis=1
            )

    new_features_count = len(feature_defs) - len(working_df.columns)
    return feature_matrix, feature_defs, max(new_features_count, 0)


def list_featuretools_primitives(primitive_type: str = "transform") -> Dict[str, Any]:
    """List all available Featuretools primitives of a given type."""
    if primitive_type == "transform":
        primitives = ft.primitives.get_transform_primitives()
    elif primitive_type == "aggregation":
        primitives = ft.primitives.get_aggregation_primitives()
    else:
        raise ValueError("primitive_type must be 'transform' or 'aggregation'.")
    return {
        "type": primitive_type,
        "count": len(primitives),
        "primitives": {
            name: {
                "name": name,
                "description": getattr(prim, "description", ""),
                "default_value": getattr(prim, "default_value", None),
            }
            for name, prim in primitives.items()
        },
    }


# ─────────────────────────────────────────────
# 7. FEATURE TRANSFORMATION TOOL
# ─────────────────────────────────────────────

@tool
def transform_features(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    method: str = "standard_scaler",
    columns: Optional[List[str]] = None,
    log_shift: float = 1.0,
    ohe_drop: str = "first",
    entity_id_col: Optional[str] = None,
    ft_agg_primitives: Optional[List[str]] = None,
    ft_trans_primitives: Optional[List[str]] = None,
    ft_max_depth: int = 1,
) -> Command:
    """
    Apply feature transformations to a DataFrame.

    Supports both classical sklearn-based scaling/encoding methods and
    automated deep feature synthesis via Featuretools.

    Parameters
    ----------
    method : str
        One of: 'standard_scaler', 'minmax_scaler', 'log', 'onehot',
        'label_encoder', 'featuretools', 'featuretools_full'.
    columns : list of str, optional
        Columns to transform. Defaults to type-appropriate columns.
    log_shift : float
        Constant added before log transform to avoid log(0). Default 1.0.
    ohe_drop : str
        OneHotEncoder drop strategy: 'first', 'if_binary', or None.
    entity_id_col : str, optional
        Column to use as the unique entity index for Featuretools.
    ft_agg_primitives : list of str, optional
        Featuretools aggregation primitives. Default [].
    ft_trans_primitives : list of str, optional
        Featuretools transformation primitives.
        Default ['add_numeric', 'multiply_numeric', 'percentile'].
    ft_max_depth : int
        Maximum depth of feature stacking in DFS. Default 1.

    Returns
    -------
    Command updating clean_df and cleaning_summary in state.
    """
    log = logging.getLogger("transform_features")
    log.info("Starting | method=%s | columns=%s", method, columns)

    df = state["clean_df"]
    result_df = df.copy()
    transformers: Dict[str, Any] = {}
    feature_defs: List[Any] = []
    new_features_count: int = 0
    cols_used: List[str] = []

    if method == "standard_scaler":
        cols_used = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols_used:
            try:
                result_df[col] = _ensure_numeric(result_df[col])
            except ValueError as e:
                log.error("standard_scaler failed on '%s': %s", col, e)
                raise
        scaler = StandardScaler()
        result_df[cols_used] = scaler.fit_transform(
            result_df[cols_used].fillna(result_df[cols_used].mean())
        )
        transformers["standard_scaler"] = scaler
        log.info("StandardScaler applied to %d columns", len(cols_used))

    elif method == "minmax_scaler":
        cols_used = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols_used:
            try:
                result_df[col] = _ensure_numeric(result_df[col])
            except ValueError as e:
                log.error("minmax_scaler failed on '%s': %s", col, e)
                raise
        scaler = MinMaxScaler()
        result_df[cols_used] = scaler.fit_transform(
            result_df[cols_used].fillna(result_df[cols_used].mean())
        )
        transformers["minmax_scaler"] = scaler
        log.info("MinMaxScaler applied to %d columns", len(cols_used))

    elif method == "log":
        cols_used = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols_used:
            try:
                result_df[col] = _ensure_numeric(result_df[col])
            except ValueError as e:
                log.error("log transform failed on '%s': %s", col, e)
                raise
            result_df[f"{col}_log"] = np.log(result_df[col].clip(lower=0) + log_shift)
        log.info("Log transform applied to %d columns", len(cols_used))

    elif method == "onehot":
        cols_used = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
        drop_param = ohe_drop if ohe_drop else None
        encoder = OneHotEncoder(drop=drop_param, sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(result_df[cols_used].astype(str))
        encoded_cols = encoder.get_feature_names_out(cols_used).tolist()
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=result_df.index)
        result_df = pd.concat([result_df.drop(columns=cols_used), encoded_df], axis=1)
        transformers["onehot_encoder"] = encoder
        log.info("OneHotEncoder: %d columns -> %d new columns", len(cols_used), len(encoded_cols))

    elif method == "label_encoder":
        cols_used = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
        label_encoders: Dict[str, LabelEncoder] = {}
        for col in cols_used:
            le = LabelEncoder()
            result_df[col] = le.fit_transform(result_df[col].astype(str))
            label_encoders[col] = le
        transformers["label_encoders"] = label_encoders
        log.info("LabelEncoder applied to %d columns", len(cols_used))

    elif method in ("featuretools", "featuretools_full"):
        result_df, feature_defs, new_features_count = _run_deep_feature_synthesis(
            df=result_df,
            columns=columns,
            entity_id_col=entity_id_col,
            agg_primitives=ft_agg_primitives or [],
            trans_primitives=ft_trans_primitives or ["add_numeric", "multiply_numeric", "percentile"],
            max_depth=ft_max_depth,
        )
        log.info("Featuretools DFS generated %d new features", new_features_count)

        if method == "featuretools_full":
            num_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                scaler = StandardScaler()
                result_df[num_cols] = scaler.fit_transform(
                    result_df[num_cols].fillna(result_df[num_cols].mean())
                )
                transformers["standard_scaler"] = scaler
                log.info("Post-DFS StandardScaler applied to %d columns", len(num_cols))
        cols_used = columns or []

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: standard_scaler, minmax_scaler, "
            "log, onehot, label_encoder, featuretools, featuretools_full."
        )

    log.info("Completed | output_shape=%s", result_df.shape)

    return Command(update={
        "clean_df": result_df,
        "cleaning_summary": {
            "transform_features": {
                "method": method,
                "transformed_columns": cols_used,
                "feature_names": result_df.columns.tolist(),
                "new_features_count": new_features_count,
                "output_shape": list(result_df.shape),
            }
        },
        "tool_priority_list_1": remove_first(state["tool_priority_list_1"]),
        "messages": [ToolMessage(
            content="transform_features completed successfully",
            tool_call_id=tool_call_id,
        )]
    })