import featuretools as ft
# import io
# import json
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
# from dateutil import parser as dateutil_parser
from rapidfuzz import fuzz, process
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from langchain_core.tools import tool

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA PROFILING TOOL
# ─────────────────────────────────────────────

# @tool







# ─────────────────────────────────────────────
# 2. MISSING VALUE HANDLING TOOL
# ─────────────────────────────────────────────

@tool
def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[List[str]] = None,
    fill_value: Optional[Any] = None,
    drop_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Detect and handle missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        One of 'mean', 'median', 'mode', 'ffill', 'bfill',
        'drop_rows', 'drop_cols', 'constant'.
    columns : list of str, optional
        Columns to apply strategy to. Defaults to all columns.
    fill_value : any, optional
        Used when strategy='constant'.
    drop_threshold : float
        For 'drop_cols', fraction of missing values above which a column is dropped.

    Returns
    -------
    Dict with cleaned DataFrame and imputation metadata.
    """
    result_df = df.copy()
    target_cols = columns if columns else df.columns.tolist()
    imputation_log: Dict[str, Any] = {}

    if strategy == "mean":
        num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
        for col in num_cols:
            missing = int(result_df[col].isnull().sum())
            if missing > 0:
                val = result_df[col].mean()
                result_df[col] = result_df[col].fillna(val)
                imputation_log[col] = {"strategy": "mean", "fill_value": float(val), "imputed_count": missing}

    elif strategy == "median":
        num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
        for col in num_cols:
            missing = int(result_df[col].isnull().sum())
            if missing > 0:
                val = result_df[col].median()
                result_df[col] = result_df[col].fillna(val)
                imputation_log[col] = {"strategy": "median", "fill_value": float(val), "imputed_count": missing}

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
            missing = int(result_df[col].isnull().sum())
            result_df[col] = result_df[col].ffill()
            imputation_log[col] = {"strategy": "ffill", "imputed_count": missing}

    elif strategy == "bfill":
        for col in target_cols:
            missing = int(result_df[col].isnull().sum())
            result_df[col] = result_df[col].bfill()
            imputation_log[col] = {"strategy": "bfill", "imputed_count": missing}

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

    return {
        "dataframe": result_df,
        "imputation_log": imputation_log,
        "remaining_missing": int(result_df.isnull().sum().sum()),
    }


# ─────────────────────────────────────────────
# 3. DUPLICATE DETECTION TOOL
# ─────────────────────────────────────────────

@tool
def detect_and_remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    remove: bool = True,
) -> Dict[str, Any]:
    """
    Detect and optionally remove duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
    subset : list of str, optional
        Columns to consider for duplicate detection.
    keep : str
        'first', 'last', or False (drop all duplicates).
    remove : bool
        Whether to remove duplicates from the returned DataFrame.

    Returns
    -------
    Dict with cleaned DataFrame and duplicate statistics.
    """
    dup_mask = df.duplicated(subset=subset, keep=False)
    n_duplicates = int(df.duplicated(subset=subset, keep=keep).sum())

    duplicate_rows = df[dup_mask].copy()

    result_df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True) if remove else df.copy()

    return {
        "dataframe": result_df,
        "statistics": {
            "original_row_count": len(df),
            "duplicate_row_count": n_duplicates,
            "rows_after_dedup": len(result_df),
            "rows_removed": len(df) - len(result_df),
            "duplicate_percentage": round(n_duplicates / max(len(df), 1) * 100, 2),
        },
        "duplicate_rows": duplicate_rows,
    }


# ─────────────────────────────────────────────
# 4. OUTLIER DETECTION TOOL
# ─────────────────────────────────────────────

@tool
def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    columns: Optional[List[str]] = None,
    action: str = "flag",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.05,
) -> Dict[str, Any]:
    """
    Detect outliers using IQR, Z-score, or Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
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
    Dict with processed DataFrame, outlier indices, and per-column counts.
    """
    result_df = df.copy()
    num_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts: Dict[str, int] = {}
    outlier_indices = set()

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
            if action == "flag":
                result_df["isolation_forest_outlier"] = 0
                result_df.loc[list(outlier_indices), "isolation_forest_outlier"] = 1

    if action == "remove" and outlier_indices:
        result_df = result_df.drop(index=list(outlier_indices)).reset_index(drop=True)

    return {
        "dataframe": result_df,
        "outlier_counts_per_column": outlier_counts,
        "total_outlier_rows": len(outlier_indices),
        "method": method,
        "action": action,
    }


# ─────────────────────────────────────────────
# 5. DATA TYPE CORRECTION TOOL
# ─────────────────────────────────────────────

@tool
def correct_data_types(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    auto_detect: bool = True,
    date_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert and correct column data types.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list of str, optional
    date_cols : list of str, optional
    categorical_cols : list of str, optional
    auto_detect : bool
        If True, attempt to auto-detect and convert likely numeric/date columns.
    date_format : str, optional
        strftime format string for date parsing.

    Returns
    -------
    Dict with corrected DataFrame and conversion log.
    """
    result_df = df.copy()
    conversion_log: Dict[str, Dict[str, str]] = {}
    errors: Dict[str, str] = {}

    def _try_numeric(col: str) -> bool:
        try:
            converted = pd.to_numeric(result_df[col], errors="coerce")
            non_null_ratio = converted.notna().sum() / max(result_df[col].notna().sum(), 1)
            if non_null_ratio >= 0.8:
                result_df[col] = converted
                conversion_log[col] = {"from": str(df[col].dtype), "to": str(result_df[col].dtype)}
                return True
        except Exception:
            pass
        return False

    def _try_date(col: str) -> bool:
        try:
            if date_format:
                converted = pd.to_datetime(result_df[col], format=date_format, errors="coerce")
            else:
                converted = pd.to_datetime(result_df[col], infer_datetime_format=True, errors="coerce")
            non_null_ratio = converted.notna().sum() / max(result_df[col].notna().sum(), 1)
            if non_null_ratio >= 0.8:
                result_df[col] = converted
                conversion_log[col] = {"from": str(df[col].dtype), "to": "datetime64"}
                return True
        except Exception:
            pass
        return False

    # Explicit conversions
    for col in (numeric_cols or []):
        try:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
            conversion_log[col] = {"from": str(df[col].dtype), "to": str(result_df[col].dtype)}
        except Exception as e:
            errors[col] = str(e)

    for col in (date_cols or []):
        try:
            if date_format:
                result_df[col] = pd.to_datetime(result_df[col], format=date_format, errors="coerce")
            else:
                result_df[col] = pd.to_datetime(result_df[col], infer_datetime_format=True, errors="coerce")
            conversion_log[col] = {"from": str(df[col].dtype), "to": "datetime64"}
        except Exception as e:
            errors[col] = str(e)

    for col in (categorical_cols or []):
        try:
            result_df[col] = result_df[col].astype("category")
            conversion_log[col] = {"from": str(df[col].dtype), "to": "category"}
        except Exception as e:
            errors[col] = str(e)

    # Auto-detect
    if auto_detect:
        skip = set((numeric_cols or []) + (date_cols or []) + (categorical_cols or []))
        for col in result_df.columns:
            if col in skip:
                continue
            if result_df[col].dtype == object:
                if not _try_date(col):
                    _try_numeric(col)

    # Detect remaining type mismatches (object cols that look numeric)
    type_issues = []
    for col in result_df.columns:
        if result_df[col].dtype == object:
            sample = result_df[col].dropna().head(20)
            numeric_like = sum(1 for v in sample if str(v).replace(".", "").replace("-", "").isdigit())
            if numeric_like / max(len(sample), 1) > 0.7:
                type_issues.append(col)

    return {
        "dataframe": result_df,
        "conversion_log": conversion_log,
        "errors": errors,
        "potential_type_issues": type_issues,
    }


# ─────────────────────────────────────────────
# 6. DATA STANDARDIZATION TOOL
# ─────────────────────────────────────────────

@tool
def standardize_data(
    df: pd.DataFrame,
    text_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    fuzzy_map: Optional[Dict[str, List[str]]] = None,
    fuzzy_threshold: int = 85,
) -> Dict[str, Any]:
    """
    Standardize text, categorical, and fuzzy-matched values.

    Parameters
    ----------
    df : pd.DataFrame
    text_cols : list of str, optional
        Columns to normalize (lowercase, strip whitespace).
    categorical_cols : list of str, optional
        Columns to standardize by mapping to canonical values.
    fuzzy_map : dict, optional
        {column: [canonical_value_1, canonical_value_2, ...]}.
        Values in the column are fuzzy-matched to the canonical list.
    fuzzy_threshold : int
        Minimum RapidFuzz score (0–100) for a match to be applied.

    Returns
    -------
    Dict with standardized DataFrame and change log.
    """
    result_df = df.copy()
    change_log: Dict[str, Any] = {}

    # Text normalization
    for col in (text_cols or []):
        if col not in result_df.columns:
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

    # Categorical standardization (to title-case / consistent format)
    for col in (categorical_cols or []):
        if col not in result_df.columns:
            continue
        before = result_df[col].copy()
        result_df[col] = result_df[col].astype(str).str.strip().str.title()
        changed = int((result_df[col] != before.astype(str)).sum())
        change_log[col] = {"operation": "categorical_standardize", "cells_changed": changed}

    # Fuzzy matching
    if fuzzy_map:
        for col, canonical_values in fuzzy_map.items():
            if col not in result_df.columns:
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

    return {"dataframe": result_df, "change_log": change_log}


# ─────────────────────────────────────────────
# 7. FEATURE TRANSFORMATION TOOL
# ─────────────────────────────────────────────


# def transform_features(
#     df: pd.DataFrame,
#     method: str = "standard_scaler",
#     columns: Optional[List[str]] = None,
#     log_shift: float = 1.0,
#     ohe_drop: str = "first",
# ) -> Dict[str, Any]:
#     """
#     Apply feature transformations to a DataFrame.

#     Parameters
#     ----------
#     df : pd.DataFrame
#     method : str
#         One of 'standard_scaler', 'minmax_scaler', 'log', 'onehot', 'label_encoder'.
#     columns : list of str, optional
#         Columns to transform. Defaults to appropriate types.
#     log_shift : float
#         Constant added before log transform to handle zeros/negatives.
#     ohe_drop : str
#         OneHotEncoder drop strategy: 'first', 'if_binary', or None.

#     Returns
#     -------
#     Dict with transformed DataFrame and fitted transformer objects.
#     """
#     result_df = df.copy()
#     transformers: Dict[str, Any] = {}

#     if method == "standard_scaler":
#         cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
#         scaler = StandardScaler()
#         result_df[cols] = scaler.fit_transform(result_df[cols].fillna(result_df[cols].mean()))
#         transformers["standard_scaler"] = scaler

#     elif method == "minmax_scaler":
#         cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
#         scaler = MinMaxScaler()
#         result_df[cols] = scaler.fit_transform(result_df[cols].fillna(result_df[cols].mean()))
#         transformers["minmax_scaler"] = scaler

#     elif method == "log":
#         cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
#         for col in cols:
#             result_df[f"{col}_log"] = np.log(result_df[col].clip(lower=0) + log_shift)

#     elif method == "onehot":
#         cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
#         drop_param = ohe_drop if ohe_drop else None
#         encoder = OneHotEncoder(drop=drop_param, sparse_output=False, handle_unknown="ignore")
#         encoded = encoder.fit_transform(result_df[cols].astype(str))
#         encoded_cols = encoder.get_feature_names_out(cols)
#         encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=result_df.index)
#         result_df = result_df.drop(columns=cols)
#         result_df = pd.concat([result_df, encoded_df], axis=1)
#         transformers["onehot_encoder"] = encoder

#     elif method == "label_encoder":
#         cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
#         label_encoders: Dict[str, LabelEncoder] = {}
#         for col in cols:
#             le = LabelEncoder()
#             result_df[col] = le.fit_transform(result_df[col].astype(str))
#             label_encoders[col] = le
#         transformers["label_encoders"] = label_encoders

#     return {
#         "dataframe": result_df,
#         "method": method,
#         "transformed_columns": columns,
#         "transformers": transformers,
#     }


# ───────────────────────────────────────────────────────────────────
# Internal helper that runs Featuretools Deep Feature Synthesis (DFS)
# ───────────────────────────────────────────────────────────────────

def _run_deep_feature_synthesis(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    entity_id_col: Optional[str],
    agg_primitives: List[str],
    trans_primitives: List[str],
    max_depth: int,
) -> Tuple[pd.DataFrame, List[Any], int]:
    """
    Internal helper that runs Featuretools Deep Feature Synthesis (DFS)
    on a single-entity dataset.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Subset of columns to include. Uses all columns if None.
    entity_id_col : str, optional
        Column to use as the unique entity index.
    agg_primitives : list of str
    trans_primitives : list of str
    max_depth : int

    Returns
    -------
    Tuple of (feature_matrix, feature_defs, new_feature_count)
    """
    working_df = df[columns].copy() if columns else df.copy()

    # Set up index
    if entity_id_col and entity_id_col in working_df.columns:
        working_df = working_df.set_index(entity_id_col)
    elif not isinstance(working_df.index, pd.RangeIndex):
        working_df = working_df.reset_index(drop=True)

    # Ensure index has a name for Featuretools
    if working_df.index.name is None:
        working_df.index.name = "id"

    # Infer logical types — cast object cols to Categorical for FT
    for col in working_df.select_dtypes(include=["object"]).columns:
        working_df[col] = working_df[col].astype("category")

    # Build Featuretools EntitySet
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

    # Run Deep Feature Synthesis
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="main",
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        max_depth=max_depth,
        verbose=False,
        n_jobs=1,
    )

    # Re-align index with original df
    feature_matrix = feature_matrix.reset_index(drop=True)

    # Merge non-targeted columns back (columns not passed to DFS)
    if columns:
        excluded_cols = [c for c in df.columns if c not in columns]
        if excluded_cols:
            feature_matrix = pd.concat(
                [df[excluded_cols].reset_index(drop=True), feature_matrix], axis=1
            )

    new_features_count = len(feature_defs) - len(working_df.columns)

    return feature_matrix, feature_defs, max(new_features_count, 0)


def list_featuretools_primitives(primitive_type: str = "transform") -> Dict[str, Any]:
    """
    List all available Featuretools primitives of a given type.

    Parameters
    ----------
    primitive_type : str
        'transform' or 'aggregation'.

    Returns
    -------
    Dict with primitive names and descriptions.
    """
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


# ─────────────────────────────────────────────────────────
# 7.5. FEATURE TRANSFORMATION TOOL (with Featuretools DFS)
# ─────────────────────────────────────────────────────────

@tool
def transform_features(
    df: pd.DataFrame,
    method: str = "standard_scaler",
    columns: Optional[List[str]] = None,
    log_shift: float = 1.0,
    ohe_drop: str = "first",
    # Featuretools-specific params
    entity_id_col: Optional[str] = None,
    ft_agg_primitives: Optional[List[str]] = None,
    ft_trans_primitives: Optional[List[str]] = None,
    ft_max_depth: int = 1,
) -> Dict[str, Any]:
    """
    Apply feature transformations to a DataFrame.

    Supports both classical sklearn-based scaling/encoding methods and
    automated deep feature synthesis via Featuretools.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to transform.
    method : str
        Transformation method. One of:
            'standard_scaler'   — Zero mean, unit variance scaling.
            'minmax_scaler'     — Scale values to [0, 1] range.
            'log'               — Log1p transformation on numeric columns.
            'onehot'            — One-hot encode categorical columns.
            'label_encoder'     — Label encode categorical columns.
            'featuretools'      — Automated feature engineering via DFS.
            'featuretools_full' — DFS + standard scaling on numeric output.
    columns : list of str, optional
        Columns to transform. Defaults to type-appropriate columns.
    log_shift : float
        Constant added before log transform to avoid log(0). Default 1.0.
    ohe_drop : str
        OneHotEncoder drop strategy: 'first', 'if_binary', or None.
    entity_id_col : str, optional
        Column to use as the unique entity index for Featuretools.
        If None, the DataFrame's existing index is used.
    ft_agg_primitives : list of str, optional
        Featuretools aggregation primitives for multi-table setups.
        Example: ['sum', 'mean', 'count', 'max', 'min'].
        Defaults to [] for single-table DFS.
    ft_trans_primitives : list of str, optional
        Featuretools transformation primitives applied per entity.
        Example: ['add_numeric', 'multiply_numeric', 'percentile', 'cum_sum'].
        Defaults to ['add_numeric', 'multiply_numeric', 'percentile'].
    ft_max_depth : int
        Maximum depth of feature stacking in DFS. Default 1.

    Returns
    -------
    Dict with keys:
        'dataframe'           — Transformed pd.DataFrame.
        'method'              — Method applied.
        'transformed_columns' — Columns targeted.
        'transformers'        — Fitted transformer objects (where applicable).
        'feature_names'       — List of output feature names.
        'new_features_count'  — Number of features generated (Featuretools only).
        'feature_defs'        — Featuretools feature definitions (DFS only).
    """
    result_df = df.copy()
    transformers: Dict[str, Any] = {}
    feature_defs: List[Any] = []
    new_features_count: int = 0

    # ── Sklearn-based methods ──────────────────────────────────────────────

    if method == "standard_scaler":
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        result_df[cols] = scaler.fit_transform(result_df[cols].fillna(result_df[cols].mean()))
        transformers["standard_scaler"] = scaler

    elif method == "minmax_scaler":
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        scaler = MinMaxScaler()
        result_df[cols] = scaler.fit_transform(result_df[cols].fillna(result_df[cols].mean()))
        transformers["minmax_scaler"] = scaler

    elif method == "log":
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            result_df[f"{col}_log"] = np.log(result_df[col].clip(lower=0) + log_shift)

    elif method == "onehot":
        cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
        drop_param = ohe_drop if ohe_drop else None
        encoder = OneHotEncoder(drop=drop_param, sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(result_df[cols].astype(str))
        encoded_cols = encoder.get_feature_names_out(cols).tolist()
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=result_df.index)
        result_df = pd.concat([result_df.drop(columns=cols), encoded_df], axis=1)
        transformers["onehot_encoder"] = encoder

    elif method == "label_encoder":
        cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
        label_encoders: Dict[str, LabelEncoder] = {}
        for col in cols:
            le = LabelEncoder()
            result_df[col] = le.fit_transform(result_df[col].astype(str))
            label_encoders[col] = le
        transformers["label_encoders"] = label_encoders

    # ── Featuretools DFS methods ───────────────────────────────────────────

    elif method in ("featuretools", "featuretools_full"):
        result_df, feature_defs, new_features_count = _run_deep_feature_synthesis(
            df=result_df,
            columns=columns,
            entity_id_col=entity_id_col,
            agg_primitives=ft_agg_primitives or [],
            trans_primitives=ft_trans_primitives or ["add_numeric", "multiply_numeric", "percentile"],
            max_depth=ft_max_depth,
        )

        # Optionally scale all numeric outputs after DFS
        if method == "featuretools_full":
            num_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                scaler = StandardScaler()
                result_df[num_cols] = scaler.fit_transform(
                    result_df[num_cols].fillna(result_df[num_cols].mean())
                )
                transformers["standard_scaler"] = scaler

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: standard_scaler, minmax_scaler, "
            "log, onehot, label_encoder, featuretools, featuretools_full."
        )

    return {
        "dataframe": result_df,
        "method": method,
        "transformed_columns": columns,
        "transformers": transformers,
        "feature_names": result_df.columns.tolist(),
        "new_features_count": new_features_count,
        "feature_defs": feature_defs,
    }




# ─────────────────────────────────────────────
# 8. DATA VALIDATION TOOL
# ─────────────────────────────────────────────

# def validate_data(
#     df: pd.DataFrame,
#     schema_dict: Optional[Dict[str, Any]] = None,
#     range_checks: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
#     type_checks: Optional[Dict[str, str]] = None,
#     not_null_cols: Optional[List[str]] = None,
# ) -> Dict[str, Any]:
#     """
#     Validate a DataFrame against schema, ranges, and type constraints.

#     Parameters
#     ----------
#     df : pd.DataFrame
#     schema_dict : dict, optional
#         Pandera-compatible column schema: {col: {'dtype': ..., 'nullable': bool, ...}}.
#     range_checks : dict, optional
#         {col: (min_val, max_val)} — pass None for unbounded side.
#     type_checks : dict, optional
#         {col: expected_dtype_string} e.g. {'age': 'int64', 'name': 'object'}.
#     not_null_cols : list of str, optional
#         Columns that must not contain nulls.

#     Returns
#     -------
#     Dict with validation results, pass/fail status, and detailed errors.
#     """
#     import pandera as pa

#     validation_results: Dict[str, Any] = {
#         "passed": True,
#         "errors": [],
#         "warnings": [],
#         "checks_run": [],
#     }

#     # Pandera schema validation
#     if schema_dict:
#         pa_cols: Dict[str, pa.Column] = {}
#         for col, config in schema_dict.items():
#             dtype = config.get("dtype", None)
#             nullable = config.get("nullable", True)
#             checks = []
#             if "min_value" in config and config["min_value"] is not None:
#                 checks.append(pa.Check.ge(config["min_value"]))
#             if "max_value" in config and config["max_value"] is not None:
#                 checks.append(pa.Check.le(config["max_value"]))
#             pa_cols[col] = pa.Column(dtype=dtype, nullable=nullable, checks=checks if checks else None)
#         try:
#             schema = pa.DataFrameSchema(pa_cols, coerce=False)
#             schema.validate(df)
#             validation_results["checks_run"].append("pandera_schema")
#         except pa.errors.SchemaError as e:
#             validation_results["passed"] = False
#             validation_results["errors"].append({"check": "pandera_schema", "detail": str(e)})

#     # Range checks
#     if range_checks:
#         for col, (min_val, max_val) in range_checks.items():
#             if col not in df.columns:
#                 validation_results["warnings"].append(f"Column '{col}' not found for range check.")
#                 continue
#             if min_val is not None:
#                 violations = int((df[col] < min_val).sum())
#                 if violations:
#                     validation_results["passed"] = False
#                     validation_results["errors"].append({
#                         "check": "range_min",
#                         "column": col,
#                         "threshold": min_val,
#                         "violations": violations,
#                     })
#             if max_val is not None:
#                 violations = int((df[col] > max_val).sum())
#                 if violations:
#                     validation_results["passed"] = False
#                     validation_results["errors"].append({
#                         "check": "range_max",
#                         "column": col,
#                         "threshold": max_val,
#                         "violations": violations,
#                     })
#         validation_results["checks_run"].append("range_checks")

#     # Type checks
#     if type_checks:
#         for col, expected_type in type_checks.items():
#             if col not in df.columns:
#                 validation_results["warnings"].append(f"Column '{col}' not found for type check.")
#                 continue
#             actual_type = str(df[col].dtype)
#             if actual_type != expected_type:
#                 validation_results["passed"] = False
#                 validation_results["errors"].append({
#                     "check": "type_mismatch",
#                     "column": col,
#                     "expected": expected_type,
#                     "actual": actual_type,
#                 })
#         validation_results["checks_run"].append("type_checks")

#     # Not-null checks
#     if not_null_cols:
#         for col in not_null_cols:
#             if col not in df.columns:
#                 validation_results["warnings"].append(f"Column '{col}' not found for null check.")
#                 continue
#             null_count = int(df[col].isnull().sum())
#             if null_count > 0:
#                 validation_results["passed"] = False
#                 validation_results["errors"].append({
#                     "check": "not_null",
#                     "column": col,
#                     "null_count": null_count,
#                 })
#         validation_results["checks_run"].append("not_null_checks")

#     return validation_results


# ─────────────────────────────────────────────
# 9. DATA VISUALIZATION TOOL
# ─────────────────────────────────────────────

# def visualize_data(
#     df: pd.DataFrame,
#     plot_type: str = "missing_heatmap",
#     columns: Optional[List[str]] = None,
#     output_path: Optional[str] = None,
#     figsize: Tuple[int, int] = (12, 6),
# ) -> Dict[str, Any]:
#     """
#     Generate data quality and distribution visualizations.

#     Parameters
#     ----------
#     df : pd.DataFrame
#     plot_type : str
#         'missing_heatmap', 'boxplot', 'histogram', 'correlation_heatmap'.
#     columns : list of str, optional
#         Columns to include in the plot.
#     output_path : str, optional
#         File path to save the plot (e.g., 'output.png').
#     figsize : tuple
#         Figure size (width, height).

#     Returns
#     -------
#     Dict with plot metadata and optional file path.
#     """
#     fig, ax = plt.subplots(figsize=figsize)
#     plot_cols = columns or df.columns.tolist()
#     metadata: Dict[str, Any] = {"plot_type": plot_type, "columns": plot_cols}

#     if plot_type == "missing_heatmap":
#         subset = df[plot_cols].isnull()
#         sns.heatmap(subset, cbar=True, yticklabels=False, cmap="viridis", ax=ax)
#         ax.set_title("Missing Values Heatmap")
#         ax.set_xlabel("Columns")
#         metadata["missing_counts"] = df[plot_cols].isnull().sum().to_dict()

#     elif plot_type == "boxplot":
#         num_cols = [c for c in plot_cols if pd.api.types.is_numeric_dtype(df[c])]
#         if not num_cols:
#             plt.close(fig)
#             return {"error": "No numeric columns found for boxplot", "plot_type": plot_type}
#         df[num_cols].plot(kind="box", ax=ax, rot=45)
#         ax.set_title("Boxplot – Outlier Detection")
#         metadata["numeric_columns"] = num_cols

#     elif plot_type == "histogram":
#         num_cols = [c for c in plot_cols if pd.api.types.is_numeric_dtype(df[c])]
#         if not num_cols:
#             plt.close(fig)
#             return {"error": "No numeric columns found for histogram", "plot_type": plot_type}
#         n = len(num_cols)
#         plt.close(fig)
#         cols_per_row = min(3, n)
#         rows = (n + cols_per_row - 1) // cols_per_row
#         fig, axes = plt.subplots(rows, cols_per_row, figsize=(figsize[0], rows * 4))
#         axes_flat = np.array(axes).flatten() if n > 1 else [axes]
#         for i, col in enumerate(num_cols):
#             axes_flat[i].hist(df[col].dropna(), bins=30, edgecolor="black", color="steelblue")
#             axes_flat[i].set_title(col)
#         for j in range(i + 1, len(axes_flat)):
#             axes_flat[j].set_visible(False)
#         fig.suptitle("Distribution Histograms")
#         plt.tight_layout()
#         metadata["numeric_columns"] = num_cols

#     elif plot_type == "correlation_heatmap":
#         num_cols = [c for c in plot_cols if pd.api.types.is_numeric_dtype(df[c])]
#         if not num_cols:
#             plt.close(fig)
#             return {"error": "No numeric columns found for correlation heatmap", "plot_type": plot_type}
#         corr = df[num_cols].corr()
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
#         ax.set_title("Correlation Heatmap")
#         metadata["correlation_matrix"] = corr.to_dict()

#     plt.tight_layout()

#     if output_path:
#         fig.savefig(output_path, dpi=150, bbox_inches="tight")
#         metadata["saved_to"] = output_path

#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
#     buf.seek(0)
#     metadata["image_bytes_size"] = len(buf.read())
#     plt.close(fig)

#     return metadata


# ─────────────────────────────────────────────
# 10. DATA CLEANING REPORT TOOL
# ─────────────────────────────────────────────

@tool
def generate_cleaning_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    operations_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generate a structured summary report comparing original and cleaned DataFrames.

    Parameters
    ----------
    original_df : pd.DataFrame
    cleaned_df : pd.DataFrame
    operations_log : list of dict, optional
        Log of operations performed (from tool outputs).

    Returns
    -------
    Dict with comprehensive cleaning report.
    """
    report: Dict[str, Any] = {}

    report["shape_comparison"] = {
        "original": {"rows": len(original_df), "columns": len(original_df.columns)},
        "cleaned": {"rows": len(cleaned_df), "columns": len(cleaned_df.columns)},
        "rows_removed": len(original_df) - len(cleaned_df),
        "columns_removed": len(original_df.columns) - len(cleaned_df.columns),
        "rows_removed_pct": round(
            (len(original_df) - len(cleaned_df)) / max(len(original_df), 1) * 100, 2
        ),
    }

    # Missing value comparison
    orig_missing = int(original_df.isnull().sum().sum())
    clean_missing = int(cleaned_df.isnull().sum().sum())
    report["missing_values"] = {
        "original_total_missing": orig_missing,
        "cleaned_total_missing": clean_missing,
        "values_imputed_or_removed": orig_missing - clean_missing,
        "missing_reduction_pct": round(
            (orig_missing - clean_missing) / max(orig_missing, 1) * 100, 2
        ),
    }

    # Duplicate comparison
    orig_dups = int(original_df.duplicated().sum())
    clean_dups = int(cleaned_df.duplicated().sum())
    report["duplicates"] = {
        "original_duplicates": orig_dups,
        "cleaned_duplicates": clean_dups,
        "duplicates_removed": orig_dups - clean_dups,
    }

    # Column changes
    added_cols = set(cleaned_df.columns) - set(original_df.columns)
    removed_cols = set(original_df.columns) - set(cleaned_df.columns)
    type_changes: Dict[str, Dict[str, str]] = {}
    for col in set(original_df.columns) & set(cleaned_df.columns):
        orig_type = str(original_df[col].dtype)
        clean_type = str(cleaned_df[col].dtype)
        if orig_type != clean_type:
            type_changes[col] = {"from": orig_type, "to": clean_type}

    report["column_changes"] = {
        "added_columns": list(added_cols),
        "removed_columns": list(removed_cols),
        "type_changes": type_changes,
    }

    # Operations log
    report["operations_performed"] = operations_log or []
    report["operations_count"] = len(operations_log or [])

    # Data quality score (simple heuristic 0–100)
    score = 100.0
    score -= min(clean_missing / max(cleaned_df.size, 1) * 100 * 2, 30)
    score -= min(clean_dups / max(len(cleaned_df), 1) * 100, 20)
    report["data_quality_score"] = round(max(score, 0), 1)

    return report


# ─────────────────────────────────────────────
# 11. FILE HANDLING TOOL
# ─────────────────────────────────────────────

# @tool






# def save_dataframe(
#     df: pd.DataFrame,
#     output_path: str,
#     file_type: Optional[str] = None,
#     index: bool = False,
#     encoding: str = "utf-8",
#     **kwargs: Any,
# ) -> Dict[str, Any]:
#     """
#     Save a DataFrame to CSV, Excel, or JSON.

#     Parameters
#     ----------
#     df : pd.DataFrame
#     output_path : str
#     file_type : str, optional
#         'csv', 'excel', 'json'. Auto-detected from extension if None.
#     index : bool
#         Whether to write row indices.
#     encoding : str
#     **kwargs :
#         Extra arguments for the pandas writer.

#     Returns
#     -------
#     Dict with save confirmation and metadata.
#     """
#     import os

#     if file_type is None:
#         ext = os.path.splitext(output_path)[-1].lower()
#         file_type = {
#             ".csv": "csv",
#             ".tsv": "csv",
#             ".xlsx": "excel",
#             ".xls": "excel",
#             ".json": "json",
#         }.get(ext, "csv")

#     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

#     if file_type == "csv":
#         df.to_csv(output_path, index=index, encoding=encoding, **kwargs)
#     elif file_type == "excel":
#         df.to_excel(output_path, index=index, **kwargs)
#     elif file_type == "json":
#         df.to_json(output_path, orient=kwargs.pop("orient", "records"), **kwargs)
#     else:
#         raise ValueError(f"Unsupported file_type: {file_type}")

#     file_size = os.path.getsize(output_path)

#     return {
#         "status": "success",
#         "output_path": output_path,
#         "file_type": file_type,
#         "rows_saved": len(df),
#         "columns_saved": len(df.columns),
#         "file_size_bytes": file_size,
#     }
