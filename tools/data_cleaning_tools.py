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
from utils.dataframe_store import load_df, save_df
from utils.serialization import to_serializable             # ← added

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_cleaning_tools")


# ─────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────

def _ensure_numeric(series: pd.Series) -> pd.Series:
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
    if pd.api.types.is_integer_dtype(series):
        return pd.to_numeric(series, downcast="integer")
    if pd.api.types.is_float_dtype(series):
        as32 = series.astype(np.float32)
        if np.allclose(series.fillna(0), as32.fillna(0), equal_nan=True, rtol=1e-5):
            return as32
    return series


def _try_bool(series: pd.Series) -> pd.Series | None:
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
    fill_value : any, optional  (required when strategy='constant')
    drop_threshold : float
    """
    log = logging.getLogger("handle_missing_values")
    log.info("Starting | strategy=%s | columns=%s", strategy, columns)

    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid options: {sorted(VALID_STRATEGIES)}")
    if strategy == "constant" and fill_value is None:
        raise ValueError("fill_value must be provided when strategy='constant'.")

    df = load_df(state["clean_df_key"])
    result_df = df.copy()
    target_cols = columns if columns else df.columns.tolist()

    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

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
                val = float(result_df[col].mean())
                result_df[col] = result_df[col].fillna(val)
                imputation_log[col] = {"strategy": "mean", "fill_value": val, "imputed_count": missing}

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
                val = float(result_df[col].median())
                result_df[col] = result_df[col].fillna(val)
                imputation_log[col] = {"strategy": "median", "fill_value": val, "imputed_count": missing}

    elif strategy == "mode":
        for col in target_cols:
            missing = int(result_df[col].isnull().sum())
            if missing > 0:
                mode_vals = result_df[col].mode()
                if len(mode_vals) > 0:
                    val = mode_vals[0]
                    result_df[col] = result_df[col].fillna(val)
                    imputation_log[col] = {"strategy": "mode", "fill_value": str(val), "imputed_count": missing}

    elif strategy == "ffill":
        for col in target_cols:
            before = int(result_df[col].isnull().sum())
            result_df[col] = result_df[col].ffill()
            after = int(result_df[col].isnull().sum())
            imputation_log[col] = {"strategy": "ffill", "imputed_count": before - after, "still_missing": after}

    elif strategy == "bfill":
        for col in target_cols:
            before = int(result_df[col].isnull().sum())
            result_df[col] = result_df[col].bfill()
            after = int(result_df[col].isnull().sum())
            imputation_log[col] = {"strategy": "bfill", "imputed_count": before - after, "still_missing": after}

    elif strategy == "constant":
        for col in target_cols:
            missing = int(result_df[col].isnull().sum())
            result_df[col] = result_df[col].fillna(fill_value)
            imputation_log[col] = {"strategy": "constant", "fill_value": str(fill_value), "imputed_count": missing}

    elif strategy == "drop_rows":
        before = len(result_df)
        result_df = result_df.dropna(subset=target_cols)
        imputation_log["dropped_rows"] = before - len(result_df)

    elif strategy == "drop_cols":
        dropped = []
        for col in target_cols:
            if result_df[col].isnull().mean() > drop_threshold:
                result_df = result_df.drop(columns=[col])
                dropped.append(col)
        imputation_log["dropped_columns"] = dropped

    remaining = int(result_df.isnull().sum().sum())
    missing_summary = {
        col: {
            "missing_count": int(result_df[col].isnull().sum()),
            "missing_pct": round(float(result_df[col].isnull().mean() * 100), 2),
        }
        for col in result_df.columns
        if result_df[col].isnull().sum() > 0
    }

    log.info("Completed | remaining_missing=%d", remaining)
    save_df(state["clean_df_key"], result_df)

    return Command(update={
        "cleaning_summary": to_serializable({
            "handle_missing_values": {
                "strategy": strategy,
                "imputation_log": imputation_log,
                "remaining_missing": remaining,
                "missing_summary": missing_summary,
            }
        }),
        "tool_priority_list_1": state["tool_priority_list_1"][1:],
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
    keep : str  — 'first', 'last', or False
    remove : bool
    """
    log = logging.getLogger("detect_and_remove_duplicates")
    df = load_df(state["clean_df_key"])
    dup_mask = df.duplicated(subset=subset, keep=False)
    n_duplicates = int(df.duplicated(subset=subset, keep=keep).sum())

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
        "duplicate_percentage": round(float(n_duplicates / max(len(df), 1) * 100), 2),
    }
    log.info("Completed | rows_removed=%d | duplicate_pct=%.2f%%",
             stats["rows_removed"], stats["duplicate_percentage"])

    save_df(state["clean_df_key"], result_df)

    return Command(update={
        "cleaning_summary": to_serializable({
            "detect_and_remove_duplicates": {
                "statistics": stats,
                "duplicate_rows_sample": duplicate_rows.head(20).to_dict(orient="records"),
            }
        }),
        "tool_priority_list_1": state["tool_priority_list_1"][1:],
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
    method : str  — 'iqr', 'zscore', or 'isolation_forest'
    columns : list of str, optional
    action : str  — 'flag', 'remove', or 'cap'
    z_threshold, iqr_multiplier, contamination : float
    """
    log = logging.getLogger("detect_outliers")
    df = load_df(state["clean_df_key"])
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
            except ValueError as e:
                log.warning("Cannot coerce '%s': %s", col, e)
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
            if action == "flag":
                result_df[f"{col}_outlier"] = mask.astype(int)
            elif action == "cap":
                result_df[col] = result_df[col].clip(lower=lower, upper=upper)

    elif method == "zscore":
        for col in num_cols:
            mean = result_df[col].mean()
            std = result_df[col].std()
            if std == 0:
                continue
            z_scores = (result_df[col] - mean) / std
            mask = z_scores.abs() > z_threshold
            outlier_counts[col] = int(mask.sum())
            outlier_indices.update(result_df.index[mask].tolist())
            if action == "flag":
                result_df[f"{col}_outlier"] = mask.astype(int)
            elif action == "cap":
                result_df[col] = result_df[col].clip(
                    lower=mean - z_threshold * std,
                    upper=mean + z_threshold * std,
                )

    elif method == "isolation_forest":
        valid_df = result_df[num_cols].dropna()
        if not valid_df.empty:
            clf = IsolationForest(contamination=contamination, random_state=42)
            preds = clf.fit_predict(valid_df)
            mask = pd.Series(preds == -1, index=valid_df.index)
            for col in num_cols:
                outlier_counts[col] = int(mask.sum())
            outlier_indices.update(valid_df.index[mask].tolist())
            if action == "flag":
                result_df["isolation_forest_outlier"] = 0
                result_df.loc[list(outlier_indices), "isolation_forest_outlier"] = 1

    if action == "remove" and outlier_indices:
        result_df = result_df.drop(index=list(outlier_indices)).reset_index(drop=True)

    log.info("Completed | total_outlier_rows=%d", len(outlier_indices))
    save_df(state["clean_df_key"], result_df)

    return Command(update={
        "cleaning_summary": to_serializable({
            "detect_outliers": {
                "method": method,
                "action": action,
                "outlier_counts_per_column": outlier_counts,
                "total_outlier_rows": len(outlier_indices),
            }
        }),
        "tool_priority_list_1": state["tool_priority_list_1"][1:],
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
    Convert and correct column data types.

    Parameters
    ----------
    numeric_cols, date_cols, categorical_cols : list of str, optional
    auto_detect : bool, default True
    date_format : str, optional
    category_threshold : float, default 0.10
    downcast_numerics : bool, default True
    inplace : bool, default False
    """
    log = logging.getLogger("fix_dtypes")
    df = load_df(state["clean_df_key"])
    out = df if inplace else df.copy()
    summary: dict[str, dict] = {}

    def _record(col: str, before: str, after: str, action: str) -> None:
        summary[col] = {"before": before, "after": after, "action": action}

    for col in (numeric_cols or []):
        if col not in out.columns:
            continue
        before = str(out[col].dtype)
        out[col] = _to_numeric(out[col])
        after = str(out[col].dtype)
        if before != after:
            _record(col, before, after, "forced numeric")

    for col in (date_cols or []):
        if col not in out.columns:
            continue
        before = str(out[col].dtype)
        out[col] = _to_datetime(out[col], date_format)
        after = str(out[col].dtype)
        if before != after:
            _record(col, before, after, "forced datetime")

    for col in (categorical_cols or []):
        if col not in out.columns:
            continue
        before = str(out[col].dtype)
        out[col] = out[col].astype("category")
        _record(col, before, "category", "forced categorical")

    if auto_detect:
        already_handled = set(
            (numeric_cols or []) + (date_cols or []) + (categorical_cols or [])
        )
        candidate_cols = [
            c for c in out.columns
            if c not in already_handled and out[c].dtype == object
        ]
        for col in candidate_cols:
            series = out[col]
            before = str(series.dtype)
            converted, action = _auto_convert(series, date_format, category_threshold)
            after = str(converted.dtype)
            if before != after:
                out[col] = converted
                _record(col, before, after, f"auto: {action}")

    if downcast_numerics:
        for col in out.select_dtypes(include=["int64", "float64"]).columns:
            before = str(out[col].dtype)
            out[col] = _downcast(out[col])
            after = str(out[col].dtype)
            if before != after:
                _record(col, before, after, "downcast")

    for col in out.select_dtypes(include="object").columns:
        if col in summary:
            continue
        converted = _try_bool(out[col])
        if converted is not None:
            before = str(out[col].dtype)
            out[col] = converted
            _record(col, before, "bool", "auto: boolean")

    still_object = out.select_dtypes(include="object").columns.tolist()
    log.info("Completed | %d columns converted", len(summary))
    save_df(state["clean_df_key"], out)

    return Command(update={
        "cleaning_summary": to_serializable({
            "fix_dtypes": {
                "conversion_summary": summary,
                "still_object_cols": still_object,
                "total_converted": len(summary),
            }
        }),
        "tool_priority_list_1": state["tool_priority_list_1"][1:],
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
    text_cols, categorical_cols : list of str, optional
    fuzzy_map : dict, optional
    fuzzy_threshold : int
    """
    log = logging.getLogger("standardize_data")
    df = load_df(state["clean_df_key"])
    result_df = df.copy()
    change_log: Dict[str, Any] = {}

    for col in (text_cols or []):
        if col not in result_df.columns:
            continue
        before = result_df[col].copy()
        result_df[col] = (
            result_df[col].astype(str).str.lower().str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        change_log[col] = {
            "operation": "text_normalize",
            "cells_changed": int((result_df[col] != before.astype(str)).sum()),
        }

    for col in (categorical_cols or []):
        if col not in result_df.columns:
            continue
        before = result_df[col].copy()
        result_df[col] = result_df[col].astype(str).str.strip().str.title()
        change_log[col] = {
            "operation": "categorical_standardize",
            "cells_changed": int((result_df[col] != before.astype(str)).sum()),
        }

    if fuzzy_map:
        for col, canonical_values in fuzzy_map.items():
            if col not in result_df.columns:
                continue
            replacements: Dict[str, str] = {}
            for val in result_df[col].dropna().unique():
                match = process.extractOne(str(val), canonical_values, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= fuzzy_threshold and str(val) != match[0]:
                    replacements[str(val)] = match[0]
            result_df[col] = result_df[col].astype(str).replace(replacements)
            change_log[f"{col}_fuzzy"] = {
                "operation": "fuzzy_match",
                "replacements": replacements,
                "cells_changed": len(replacements),
            }

    log.info("Completed | %d columns processed", len(change_log))
    save_df(state["clean_df_key"], result_df)

    return Command(update={
        "cleaning_summary": to_serializable({
            "standardize_data": {
                "change_log": change_log,
                "total_columns_processed": len(change_log),
            }
        }),
        "tool_priority_list_1": state["tool_priority_list_1"][1:],
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

    Parameters
    ----------
    method : str
        One of: 'standard_scaler', 'minmax_scaler', 'log', 'onehot',
        'label_encoder', 'featuretools', 'featuretools_full'.
    columns, log_shift, ohe_drop, entity_id_col,
    ft_agg_primitives, ft_trans_primitives, ft_max_depth
    """
    log = logging.getLogger("transform_features")
    df = load_df(state["clean_df_key"])
    result_df = df.copy()
    new_features_count: int = 0
    cols_used: List[str] = []

    if method == "standard_scaler":
        cols_used = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols_used:
            result_df[col] = _ensure_numeric(result_df[col])
        scaler = StandardScaler()
        result_df[cols_used] = scaler.fit_transform(
            result_df[cols_used].fillna(result_df[cols_used].mean())
        )

    elif method == "minmax_scaler":
        cols_used = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols_used:
            result_df[col] = _ensure_numeric(result_df[col])
        scaler = MinMaxScaler()
        result_df[cols_used] = scaler.fit_transform(
            result_df[cols_used].fillna(result_df[cols_used].mean())
        )

    elif method == "log":
        cols_used = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols_used:
            result_df[col] = _ensure_numeric(result_df[col])
            result_df[f"{col}_log"] = np.log(result_df[col].clip(lower=0) + log_shift)

    elif method == "onehot":
        cols_used = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
        drop_param = ohe_drop if ohe_drop else None
        encoder = OneHotEncoder(drop=drop_param, sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(result_df[cols_used].astype(str))
        encoded_cols = encoder.get_feature_names_out(cols_used).tolist()
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=result_df.index)
        result_df = pd.concat([result_df.drop(columns=cols_used), encoded_df], axis=1)

    elif method == "label_encoder":
        cols_used = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cols_used:
            le = LabelEncoder()
            result_df[col] = le.fit_transform(result_df[col].astype(str))

    elif method in ("featuretools", "featuretools_full"):
        result_df, feature_defs, new_features_count = _run_deep_feature_synthesis(
            df=result_df,
            columns=columns,
            entity_id_col=entity_id_col,
            agg_primitives=ft_agg_primitives or [],
            trans_primitives=ft_trans_primitives or ["add_numeric", "multiply_numeric", "percentile"],
            max_depth=ft_max_depth,
        )
        if method == "featuretools_full":
            num_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                scaler = StandardScaler()
                result_df[num_cols] = scaler.fit_transform(
                    result_df[num_cols].fillna(result_df[num_cols].mean())
                )
        cols_used = columns or []

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: standard_scaler, minmax_scaler, "
            "log, onehot, label_encoder, featuretools, featuretools_full."
        )

    log.info("Completed | output_shape=%s", result_df.shape)
    save_df(state["clean_df_key"], result_df)

    return Command(update={
        "cleaning_summary": to_serializable({
            "transform_features": {
                "method": method,
                "transformed_columns": cols_used,
                "feature_names": result_df.columns.tolist(),
                "new_features_count": new_features_count,
                "output_shape": list(result_df.shape),
            }
        }),
        "tool_priority_list_1": state["tool_priority_list_1"][1:],
        "messages": [ToolMessage(
            content="transform_features completed successfully",
            tool_call_id=tool_call_id,
        )]
    })