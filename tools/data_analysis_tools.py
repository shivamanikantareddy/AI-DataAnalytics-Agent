"""
================================================================================
Enterprise Analytics Toolkit for AI Data Analysis Agent
================================================================================
"""

from __future__ import annotations

import warnings
from typing import Any, Annotated
from utils.state import AgentState
from utils.dataframe_store import load_df
from utils.serialization import to_serializable              # ← added
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, pointbiserialr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# ============================================================
# MODULE 1 — STATISTICAL SUMMARY ANALYSIS
# ============================================================

@tool
def characterize_distributions(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    columns: list[str] | None = None,
    skew_threshold: float = 0.5,
    kurt_threshold: float = 1.0,
) -> Command:
    """
    Characterize the statistical distribution of numeric columns beyond .describe().

    Args:
        columns: List of numeric columns to analyze. Defaults to all numeric cols.
        skew_threshold: Absolute skewness above which a column is flagged as skewed.
        kurt_threshold: Excess kurtosis above which a column is flagged as heavy-tailed.
    """
    df = load_df(state["clean_df_key"])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    target_cols = [c for c in (columns or num_cols) if c in num_cols]

    results = {}

    for col in target_cols:
        series = df[col].dropna()
        if len(series) < 4:
            results[col] = {"error": "insufficient data"}
            continue

        skew = float(series.skew())
        kurt = float(series.kurt())
        mean = float(series.mean())
        std = float(series.std())
        p25 = float(series.quantile(0.25))
        p50 = float(series.median())
        p75 = float(series.quantile(0.75))
        iqr = p75 - p25

        if abs(skew) < skew_threshold:
            shape = "approximately_normal"
        elif skew > skew_threshold:
            shape = "right_skewed"
        else:
            shape = "left_skewed"

        if iqr > 0 and (std / iqr) > 2.0 and abs(skew) < 0.3:
            shape = "bimodal_hint"

        if kurt > kurt_threshold:
            tail = "heavy_tailed"
        elif kurt < -kurt_threshold:
            tail = "light_tailed"
        else:
            tail = "normal_tails"

        cv = (std / mean) if mean != 0 else None
        zero_pct = float((series == 0).mean() * 100)

        notes = []
        if shape == "right_skewed":
            notes.append("Consider log-transform for modeling/visualization.")
        if shape == "bimodal_hint":
            notes.append("Possible bimodal distribution; investigate segment-level splits.")
        if tail == "heavy_tailed":
            notes.append("Heavy tails suggest extreme values; outlier analysis recommended.")
        if zero_pct > 20:
            notes.append(f"{zero_pct:.1f}% zeros detected; may indicate sparse/inactive records.")

        results[col] = {
            "mean": round(mean, 4),
            "median": round(p50, 4),
            "std": round(std, 4),
            "min": float(series.min()),
            "max": float(series.max()),
            "range": float(series.max() - series.min()),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "distribution_shape": shape,
            "tail_behavior": tail,
            "dominant_range": [round(p25, 4), round(p75, 4)],
            "iqr": round(iqr, 4),
            "cv": round(cv, 4) if cv is not None else None,
            "zero_pct": round(zero_pct, 2),
            "n_valid": int(len(series)),
            "recommendation": " ".join(notes) if notes else "Distribution appears standard.",
        }

    return Command(update={
        "analysis_results": to_serializable({"characterize_distributions": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="characterize_distributions completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def detect_variance_anomalies(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    group_col: str | None = None,
    columns: list[str] | None = None,
    cv_flag_threshold: float = 1.5,
) -> Command:
    """
    Detect columns with unusually high or low variance (relative to their mean).

    Args:
        group_col: Optional categorical column to segment variance analysis.
        columns: Numeric columns to analyze. Defaults to all numeric.
        cv_flag_threshold: CV above this value is flagged as high-variance.
    """
    df = load_df(state["clean_df_key"])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    target_cols = [c for c in (columns or num_cols) if c in num_cols]

    col_stats: dict[str, dict] = {}
    high_var, low_var = [], []

    for col in target_cols:
        series = df[col].dropna()
        mean = float(series.mean())
        std = float(series.std())
        cv = abs(std / mean) if mean != 0 else None

        col_stats[col] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "cv": round(float(cv), 4) if cv is not None else None,
        }

        if cv is not None:
            if cv > cv_flag_threshold:
                high_var.append(col)
            elif cv < 0.01:
                low_var.append(col)
        elif std < 1e-8:
            low_var.append(col)

    group_variance: dict[str, Any] = {}
    if group_col and group_col in df.columns:
        for col in target_cols:
            group_variance[col] = (
                df.groupby(group_col)[col]
                .agg(["mean", "std"])
                .round(4)
                .to_dict(orient="index")
            )

    results = {
        "high_variance_cols": high_var,
        "low_variance_cols": low_var,
        "col_stats": col_stats,
        "group_variance": group_variance,
    }

    return Command(update={
        "analysis_results": to_serializable({"detect_variance_anomalies": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="detect_variance_anomalies completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ============================================================
# MODULE 2 — CORRELATION & RELATIONSHIP DISCOVERY
# ============================================================

@tool
def compute_correlation_matrix(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    columns: list[str] | None = None,
    method: str = "pearson",
    strong_threshold: float = 0.6,
) -> Command:
    """
    Compute a full correlation matrix and extract actionable relationship insights.

    Args:
        columns: Columns to include. Defaults to all numeric.
        method: "pearson" (linear) or "spearman" (monotonic/nonlinear).
        strong_threshold: Absolute correlation above this is flagged as strong.
    """
    df = load_df(state["clean_df_key"])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    target_cols = [c for c in (columns or num_cols) if c in num_cols]

    if len(target_cols) < 2:
        return Command(update={
            "analysis_results": to_serializable({"compute_correlation_matrix_result": {"error": "Need at least 2 numeric columns."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content="Error: need at least 2 numeric columns.", tool_call_id=tool_call_id)]
        })

    sub = df[target_cols].dropna()
    corr_matrix = sub.corr(method=method)

    strong_pairs = []
    redundant_pairs = []
    driver_score: dict[str, int] = {c: 0 for c in target_cols}

    for i, col_a in enumerate(target_cols):
        for col_b in target_cols[i + 1:]:
            r = corr_matrix.loc[col_a, col_b]
            if pd.isna(r):
                continue
            abs_r = abs(r)

            if abs_r >= strong_threshold:
                direction = "positive" if r > 0 else "negative"
                strong_pairs.append({
                    "col_a": col_a,
                    "col_b": col_b,
                    "correlation": round(float(r), 4),
                    "direction": direction,
                    "interpretation": f"Strong {direction} relationship between {col_a} and {col_b}.",
                })
                driver_score[col_a] += 1
                driver_score[col_b] += 1

            if abs_r >= 0.9:
                redundant_pairs.append({
                    "col_a": col_a,
                    "col_b": col_b,
                    "correlation": round(float(r), 4),
                    "risk": "multicollinearity",
                })

    driver_candidates = sorted(driver_score.items(), key=lambda x: x[1], reverse=True)
    driver_candidates = [
        {"column": c, "strong_correlation_count": n}
        for c, n in driver_candidates if n > 0
    ]

    results = {
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "strong_pairs": strong_pairs,
        "redundant_pairs": redundant_pairs,
        "driver_candidates": driver_candidates,
        "method_used": method,
        "n_columns_analyzed": int(len(target_cols)),
    }

    return Command(update={
        "analysis_results": to_serializable({"compute_correlation_matrix_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="compute_correlation_matrix completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def detect_nonlinear_relationships(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    target_col: str,
    feature_cols: list[str] | None = None,
    pearson_nonlinear_gap: float = 0.2,
) -> Command:
    """
    Identify features with nonlinear relationships to a target variable.

    Args:
        target_col: The column treated as the outcome/target.
        feature_cols: Predictor columns to test. Defaults to all other numeric cols.
        pearson_nonlinear_gap: If |spearman - pearson| > this, flag as nonlinear.
    """
    df = load_df(state["clean_df_key"])
    if target_col not in df.columns:
        return Command(update={
            "analysis_results": to_serializable({"detect_nonlinear_relationships_result": {"error": f"Target column '{target_col}' not found."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: '{target_col}' not found.", tool_call_id=tool_call_id)]
        })

    num_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in (feature_cols or num_cols) if c in num_cols and c != target_col]

    comparison: list[dict] = []
    nonlinear_candidates: list[dict] = []

    for col in feature_cols:
        sub = df[[col, target_col]].dropna()
        if len(sub) < 10:
            continue

        pearson_r, pearson_p = stats.pearsonr(sub[col], sub[target_col])
        spearman_r, spearman_p = spearmanr(sub[col], sub[target_col])
        gap = abs(float(spearman_r) - float(pearson_r))

        entry = {
            "feature": col,
            "pearson_r": round(float(pearson_r), 4),
            "spearman_r": round(float(spearman_r), 4),
            "gap": round(gap, 4),
            "pearson_pvalue": round(float(pearson_p), 5),
            "spearman_pvalue": round(float(spearman_p), 5),
        }
        comparison.append(entry)

        if gap >= pearson_nonlinear_gap and abs(float(spearman_r)) > 0.3:
            entry["nonlinear_hint"] = (
                f"{col} shows stronger monotonic than linear association with {target_col}. "
                "Consider log/sqrt transform or use Spearman-based visualization."
            )
            nonlinear_candidates.append(entry)

    comparison.sort(key=lambda x: abs(x["spearman_r"]), reverse=True)

    results = {
        "target": target_col,
        "nonlinear_candidates": nonlinear_candidates,
        "comparison_table": comparison,
        "n_features_tested": int(len(comparison)),
    }

    return Command(update={
        "analysis_results": to_serializable({"detect_nonlinear_relationships_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="detect_nonlinear_relationships completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ============================================================
# MODULE 3 — FEATURE IMPORTANCE / DRIVER ANALYSIS
# ============================================================

@tool
def compute_feature_importance(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    target_col: str,
    feature_cols: list[str] | None = None,
    method: str = "random_forest",
    top_n: int = 10,
) -> Command:
    """
    Estimate feature importance scores to identify key drivers of a target metric.

    Args:
        target_col: The business metric to explain (e.g., "revenue").
        feature_cols: Predictor columns. Defaults to all other columns.
        method: "random_forest" or "linear".
        top_n: Number of top drivers to return.
    """
    df = load_df(state["clean_df_key"])

    if target_col not in df.columns:
        return Command(update={
            "analysis_results": to_serializable({"compute_feature_importance_result": {"error": f"Target '{target_col}' not found."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: '{target_col}' not found.", tool_call_id=tool_call_id)]
        })

    all_cols = [c for c in df.columns if c != target_col]
    feature_cols = feature_cols or all_cols

    work = df[feature_cols + [target_col]].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[target_col])

    notes = []
    for col in feature_cols:
        if col not in work.columns:
            continue
        if work[col].dtype == object or str(work[col].dtype) == "category":
            try:
                le = LabelEncoder()
                work[col] = le.fit_transform(work[col].astype(str))
                notes.append(f"{col}: label-encoded for importance analysis.")
            except Exception:
                work.drop(columns=[col], inplace=True)

    valid_features = [c for c in feature_cols if c in work.columns]
    X = work[valid_features].replace([np.inf, -np.inf], np.nan)
    col_medians = X.median().fillna(0)
    X = X.fillna(col_medians)
    float32_max = float(np.finfo(np.float32).max)
    X = X.clip(-float32_max, float32_max)

    y = work[target_col].replace([np.inf, -np.inf], np.nan)
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if X.empty or len(X) < 10:
        return Command(update={
            "analysis_results": to_serializable({"compute_feature_importance_result": {"error": "Insufficient data after preprocessing."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content="Error: insufficient data.", tool_call_id=tool_call_id)]
        })

    importances: list[tuple[str, float]] = []

    if method == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X, y)
        importances = list(zip(valid_features, model.feature_importances_))
    elif method == "linear":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        importances = list(zip(valid_features, np.abs(model.coef_)))
    else:
        return Command(update={
            "analysis_results": to_serializable({"compute_feature_importance_result": {"error": f"Unknown method '{method}'."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: unknown method '{method}'.", tool_call_id=tool_call_id)]
        })

    importances.sort(key=lambda x: x[1], reverse=True)
    top = importances[:top_n]
    total = sum(v for _, v in top) or 1.0

    top_drivers = [
        {
            "rank": i + 1,
            "feature": feat,
            "importance_score": round(float(val), 6),
            "relative_importance_pct": round(float(val) / total * 100, 2),
        }
        for i, (feat, val) in enumerate(top)
    ]

    results = {
        "target_col": target_col,
        "method_used": method,
        "top_drivers": top_drivers,
        "n_features_evaluated": int(len(valid_features)),
        "notes": notes,
    }

    return Command(update={
        "analysis_results": to_serializable({"compute_feature_importance_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="compute_feature_importance completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def compute_variance_contribution(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    target_col: str,
    group_cols: list[str],
) -> Command:
    """
    Quantify how much of the target variable's variance is explained by each
    categorical grouping variable (eta-squared / ANOVA-based decomposition).

    Args:
        target_col: Numeric outcome column.
        group_cols: Categorical columns to test as grouping factors.
    """
    df = load_df(state["clean_df_key"])
    if target_col not in df.columns:
        return Command(update={
            "analysis_results": to_serializable({"compute_variance_contribution_result": {"error": f"Target '{target_col}' not found."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: '{target_col}' not found.", tool_call_id=tool_call_id)]
        })

    results = []
    for gcol in group_cols:
        if gcol not in df.columns:
            continue
        sub = df[[gcol, target_col]].dropna()
        groups = [grp[target_col].values for _, grp in sub.groupby(gcol) if len(grp) > 1]
        if len(groups) < 2:
            continue
        try:
            kruskal_stat, kruskal_p = kruskal(*groups)
        except ValueError:
            continue

        grand_mean = float(sub[target_col].mean())
        ss_between = sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups)
        ss_total = float(((sub[target_col] - grand_mean) ** 2).sum())
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

        interp = (
            "strong group effect" if eta_sq > 0.14
            else "moderate group effect" if eta_sq > 0.06
            else "small or negligible group effect"
        )

        results.append({
            "group_col": gcol,
            "eta_squared": round(eta_sq, 4),
            "kruskal_pvalue": round(float(kruskal_p), 6),
            "n_groups": int(len(groups)),
            "interpretation": interp,
            "statistically_significant": bool(kruskal_p < 0.05),
        })

    results.sort(key=lambda x: x["eta_squared"], reverse=True)

    final_results = {
        "target_col": target_col,
        "results": results,
        "top_driver": results[0]["group_col"] if results else None,
    }

    return Command(update={
        "analysis_results": to_serializable({"compute_variance_contribution_result": final_results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="compute_variance_contribution completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ============================================================
# MODULE 4 — OUTLIER & ANOMALY DETECTION
# ============================================================

@tool
def detect_statistical_outliers(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    columns: list[str] | None = None,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    return_rows: bool = True,
) -> Command:
    """
    Detect outliers in numeric columns using IQR fence or Z-score methods.

    Args:
        columns: Columns to check. Defaults to all numeric.
        method: "iqr" or "zscore".
        iqr_multiplier: Fence multiplier.
        zscore_threshold: Z-score above which a value is an outlier.
        return_rows: Whether to include actual outlier rows in output.
    """
    df = load_df(state["clean_df_key"])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    target_cols = [c for c in (columns or num_cols) if c in num_cols]

    per_column: dict[str, Any] = {}
    summary_rows: list[dict] = []

    for col in target_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue

        if method == "iqr":
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask = (df[col] < lower) | (df[col] > upper)
            fence_info = {"lower_fence": round(lower, 4), "upper_fence": round(upper, 4)}
        else:
            z_scores = np.abs(stats.zscore(series))
            outlier_idx_in_series = series.index[z_scores > zscore_threshold]
            mask = df.index.isin(outlier_idx_in_series)
            fence_info = {"zscore_threshold": float(zscore_threshold)}

        outlier_count = int(mask.sum())
        outlier_pct = round(float(outlier_count / len(df) * 100), 2)

        result: dict[str, Any] = {"outlier_count": outlier_count, "outlier_pct": outlier_pct, **fence_info}

        if return_rows and outlier_count > 0:
            result["outlier_indices"] = [int(i) for i in df.index[mask].tolist()]
            result["outlier_values"] = [float(v) for v in df.loc[mask, col].round(4).tolist()]

        per_column[col] = result
        summary_rows.append({"column": col, "outlier_pct": outlier_pct, "count": outlier_count})

    summary_rows.sort(key=lambda x: x["outlier_pct"], reverse=True)

    results = {
        "method": method,
        "per_column": per_column,
        "summary_ranked": summary_rows,
        "high_outlier_cols": [r["column"] for r in summary_rows if r["outlier_pct"] > 5.0],
    }

    return Command(update={
        "analysis_results": to_serializable({"detect_statistical_outliers_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="detect_statistical_outliers completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def detect_rare_categories(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    columns: list[str] | None = None,
    rare_threshold_pct: float = 2.0,
) -> Command:
    """
    Identify rare or sparse categories in categorical columns.

    Args:
        columns: Categorical columns to analyze. Defaults to all object/category cols.
        rare_threshold_pct: Categories appearing less than this % of total are "rare".
    """
    df = load_df(state["clean_df_key"])
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    target_cols = [c for c in (columns or cat_cols) if c in cat_cols]

    results: dict[str, Any] = {}

    for col in target_cols:
        counts = df[col].value_counts(dropna=True)
        total = int(counts.sum())
        if total == 0:
            continue

        pcts = (counts / total * 100).round(2)
        rare = [
            {"value": str(v), "count": int(counts[v]), "pct": float(pcts[v])}
            for v in counts.index if float(pcts[v]) < rare_threshold_pct
        ]

        dominant = {"value": str(counts.index[0]), "pct": float(pcts.iloc[0])}
        imbalance = round(float(pcts.iloc[0] / pcts.iloc[1]), 2) if len(pcts) > 1 else None

        results[col] = {
            "total_categories": int(len(counts)),
            "rare_categories": rare,
            "rare_count": int(len(rare)),
            "dominant_category": dominant,
            "imbalance_ratio": imbalance,
            "recommendation": (
                f"{len(rare)} rare categories detected. Consider grouping into 'Other'."
                if rare else "No rare categories detected."
            ),
        }

    return Command(update={
        "analysis_results": to_serializable({"detect_rare_categories_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="detect_rare_categories completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def detect_metric_spikes(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    value_col: str,
    time_col: str | None = None,
    group_col: str | None = None,
    spike_zscore: float = 2.5,
) -> Command:
    """
    Detect sudden spikes or drops in a business metric, optionally across time or groups.

    Args:
        value_col: The metric column to monitor.
        time_col: Optional time column to sort by before spike detection.
        group_col: Optional group column to detect spikes within each group.
        spike_zscore: Z-score threshold for spike flagging.
    """
    df = load_df(state["clean_df_key"])
    if value_col not in df.columns:
        return Command(update={
            "analysis_results": to_serializable({"detect_metric_spikes_result": {"error": f"Column '{value_col}' not found."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: '{value_col}' not found.", tool_call_id=tool_call_id)]
        })

    work = df.copy()
    if time_col and time_col in work.columns:
        work = work.sort_values(time_col)

    spike_rows: list[dict] = []

    def _find_spikes(sub: pd.DataFrame, group_label: str | None = None) -> None:
        series = sub[value_col].dropna()
        if len(series) < 4:
            return
        z = (series - series.mean()) / series.std()
        flagged = z[np.abs(z) > spike_zscore]
        for idx, z_val in flagged.items():
            spike_rows.append({
                "index": int(idx),
                "value": round(float(sub.loc[idx, value_col]), 4),
                "zscore": round(float(z_val), 4),
                "direction": "spike_up" if z_val > 0 else "spike_down",
                "group": group_label,
            })

    if group_col and group_col in work.columns:
        for grp, sub_df in work.groupby(group_col):
            _find_spikes(sub_df, group_label=str(grp))
    else:
        _find_spikes(work)

    spike_rows.sort(key=lambda x: abs(x["zscore"]), reverse=True)

    severity_summary = {
        "extreme_spikes": int(sum(1 for s in spike_rows if abs(s["zscore"]) > spike_zscore * 1.5)),
        "moderate_spikes": int(sum(1 for s in spike_rows if abs(s["zscore"]) <= spike_zscore * 1.5)),
        "spike_directions": {
            "up": int(sum(1 for s in spike_rows if s["direction"] == "spike_up")),
            "down": int(sum(1 for s in spike_rows if s["direction"] == "spike_down")),
        },
    }

    results = {
        "value_col": value_col,
        "spike_rows": spike_rows[:50],
        "n_spikes": int(len(spike_rows)),
        "severity_summary": severity_summary,
    }

    return Command(update={
        "analysis_results": to_serializable({"detect_metric_spikes_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="detect_metric_spikes completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ============================================================
# MODULE 5 — SEGMENTATION & CLUSTERING
# ============================================================

@tool
def cluster_companies(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    feature_cols: list[str],
    n_clusters: int | None = None,
    max_clusters: int = 6,
    cluster_label_col: str = "cluster_label",
    scale: bool = True,
) -> Command:
    """
    Segment companies (or rows) into clusters using KMeans.

    Args:
        feature_cols: Numeric features to cluster on.
        n_clusters: Number of clusters. If None, auto-selected via silhouette score.
        max_clusters: Maximum K to evaluate in auto-selection.
        cluster_label_col: Name of the cluster assignment column.
        scale: Whether to StandardScale features before clustering.
    """
    df = load_df(state["clean_df_key"])
    valid_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if len(valid_cols) < 2:
        return Command(update={
            "analysis_results": to_serializable({"cluster_companies_result": {"error": "Need at least 2 valid numeric feature columns."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content="Error: need >= 2 numeric columns.", tool_call_id=tool_call_id)]
        })

    work = df[valid_cols].dropna()
    if len(work) < 10:
        return Command(update={
            "analysis_results": to_serializable({"cluster_companies_result": {"error": "Insufficient rows for clustering (need >= 10)."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content="Error: insufficient rows.", tool_call_id=tool_call_id)]
        })

    X = work.values
    X_scaled = StandardScaler().fit_transform(X) if scale else X

    if n_clusters is None:
        best_k, best_score = 2, -1.0
        for k in range(2, min(max_clusters + 1, len(work))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = float(silhouette_score(X_scaled, labels))
            if score > best_score:
                best_k, best_score = k, score
        n_clusters = best_k
        auto_selected = True
    else:
        auto_selected = False

    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km_final.fit_predict(X_scaled)
    sil_score = float(silhouette_score(X_scaled, cluster_ids))

    result_df = work.copy()
    result_df[cluster_label_col] = cluster_ids
    profiles = result_df.groupby(cluster_label_col)[valid_cols].mean().round(4)
    sizes = {int(k): int(v) for k, v in result_df[cluster_label_col].value_counts().sort_index().items()}

    overall_means = work[valid_cols].mean()
    descriptions: dict[str, str] = {}
    for cid in sorted(profiles.index):
        traits = []
        for col in valid_cols:
            cluster_val = float(profiles.loc[cid, col])
            overall_val = float(overall_means[col])
            if overall_val != 0:
                delta_pct = (cluster_val - overall_val) / abs(overall_val) * 100
                if delta_pct > 25:
                    traits.append(f"high {col} (+{delta_pct:.0f}%)")
                elif delta_pct < -25:
                    traits.append(f"low {col} ({delta_pct:.0f}%)")
        descriptions[f"cluster_{int(cid)}"] = (
            "Characterized by: " + ", ".join(traits) if traits else "Near-average profile."
        )

    results = {
        "n_clusters": int(n_clusters),
        "auto_selected_k": bool(auto_selected),
        "silhouette_score": round(sil_score, 4),
        "cluster_sizes": sizes,
        "cluster_profiles": {int(k): {c: float(v) for c, v in row.items()} for k, row in profiles.to_dict(orient="index").items()},
        "cluster_descriptions": descriptions,
        "cluster_assignments": {int(k): int(v) for k, v in zip(work.index.tolist(), cluster_ids.tolist())},
        "features_used": valid_cols,
    }

    return Command(update={
        "analysis_results": to_serializable({"cluster_companies_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="cluster_companies completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def segment_by_quantile(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    value_col: str,
    n_tiers: int = 4,
    tier_labels: list[str] | None = None,
) -> Command:
    """
    Segment records into performance tiers based on quantile cut of a single metric.

    Args:
        value_col: Column to tier on (e.g., "revenue", "growth_rate").
        n_tiers: Number of tiers (4 = quartile, 5 = quintile, etc.).
        tier_labels: Custom labels. Defaults to ["Tier 1 (Bottom)", ..., "Tier N (Top)"].
    """
    df = load_df(state["clean_df_key"])

    if value_col not in df.columns:
        return Command(update={
            "analysis_results": to_serializable({"segment_by_quantile_result": {"error": f"Column '{value_col}' not found."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: '{value_col}' not found.", tool_call_id=tool_call_id)]
        })

    if tier_labels is None:
        tier_labels = [f"Tier {i+1}" for i in range(n_tiers)]
        tier_labels[-1] += " (Top)"
        tier_labels[0] += " (Bottom)"

    if len(tier_labels) != n_tiers:
        return Command(update={
            "analysis_results": to_serializable({"segment_by_quantile_result": {"error": "tier_labels length must match n_tiers."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content="Error: tier_labels length mismatch.", tool_call_id=tool_call_id)]
        })

    work = df.copy()

    if not pd.api.types.is_numeric_dtype(work[value_col]):
        work[value_col] = pd.to_numeric(
            work[value_col].astype(str).str.strip()
            .str.replace(r"[$€£¥₹%,\s]", "", regex=True)
            .str.replace(r"[^\d.\-eE]", "", regex=True),
            errors="coerce",
        )
        coerced_nulls = int(work[value_col].isna().sum())

        if work[value_col].dropna().empty:
            return Command(update={
                "analysis_results": to_serializable({"segment_by_quantile_result": {"error": f"Column '{value_col}' has no numeric values after coercion."}}),
                "tool_priority_list_2": state["tool_priority_list_2"][1:],
                "messages": [ToolMessage(content=f"Error: '{value_col}' not numeric.", tool_call_id=tool_call_id)]
            })

        if coerced_nulls > len(work) * 0.3:
            return Command(update={
                "analysis_results": to_serializable({"segment_by_quantile_result": {"error": f"Column '{value_col}' produced too many NaNs after coercion."}}),
                "tool_priority_list_2": state["tool_priority_list_2"][1:],
                "messages": [ToolMessage(content=f"Error: '{value_col}' coercion produced too many NaNs.", tool_call_id=tool_call_id)]
            })

    try:
        work["_tier_"] = pd.qcut(work[value_col], q=n_tiers, labels=tier_labels, duplicates="drop")
    except ValueError as e:
        return Command(update={
            "analysis_results": to_serializable({"segment_by_quantile_result": {"error": f"Could not create tiers: {e}"}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: could not create tiers: {e}", tool_call_id=tool_call_id)]
        })

    tier_dist = (
        work["_tier_"].value_counts().rename_axis("tier").reset_index(name="count")
    )
    tier_dist["pct"] = (tier_dist["count"] / len(work) * 100).round(2)

    tier_boundaries: dict[str, dict] = {}
    for label in tier_labels:
        sub = work[work["_tier_"] == label][value_col].dropna()
        if len(sub):
            tier_boundaries[label] = {
                "min": round(float(sub.min()), 4),
                "max": round(float(sub.max()), 4),
                "mean": round(float(sub.mean()), 4),
            }

    num_cols = work.select_dtypes(include="number").columns.tolist()
    tier_profiles_raw = work.groupby("_tier_", observed=True)[num_cols].mean().round(4).to_dict(orient="index")
    tier_profiles = {str(k): {c: float(v) for c, v in row.items()} for k, row in tier_profiles_raw.items()}
    tier_assignments = {str(idx): str(val) for idx, val in work["_tier_"].items() if pd.notna(val)}

    results = {
        "value_col": value_col,
        "n_tiers": int(n_tiers),
        "tier_distribution": tier_dist.to_dict(orient="records"),
        "tier_boundaries": tier_boundaries,
        "tier_profiles": tier_profiles,
        "tier_assignments": tier_assignments,
    }

    return Command(update={
        "analysis_results": to_serializable({"segment_by_quantile_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="segment_by_quantile completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ============================================================
# MODULE 6 — TREND & TIME-SERIES INSIGHT DETECTION
# ============================================================

@tool
def detect_time_trends(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    time_col: str,
    value_cols: list[str],
    freq: str = "auto",
    trend_pvalue_threshold: float = 0.05,
) -> Command:
    """
    Detect trends, growth/decline patterns, and acceleration/deceleration in time series.

    Args:
        time_col: Name of the datetime or sortable time column.
        value_cols: Numeric columns to analyze over time.
        freq: Resampling frequency ("auto", "M", "Q", "Y").
        trend_pvalue_threshold: OLS p-value below which trend is considered significant.
    """
    df = load_df(state["clean_df_key"])
    if time_col not in df.columns:
        return Command(update={
            "analysis_results": to_serializable({"detect_time_trends_result": {"error": f"Time column '{time_col}' not found."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: '{time_col}' not found.", tool_call_id=tool_call_id)]
        })

    work = df.copy()
    try:
        work[time_col] = pd.to_datetime(work[time_col])
    except Exception:
        return Command(update={
            "analysis_results": to_serializable({"detect_time_trends_result": {"error": f"Could not parse '{time_col}' as datetime."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: could not parse '{time_col}'.", tool_call_id=tool_call_id)]
        })

    work = work.sort_values(time_col).dropna(subset=[time_col])
    valid_value_cols = [c for c in value_cols if c in work.columns and pd.api.types.is_numeric_dtype(work[c])]

    if freq != "auto":
        work = work.set_index(time_col)[valid_value_cols].resample(freq).mean().reset_index()

    time_numeric = (work[time_col] - work[time_col].min()).dt.days.values
    col_results: dict[str, Any] = {}

    for col in valid_value_cols:
        series = work[col].dropna()
        if len(series) < 4:
            col_results[col] = {"error": "Insufficient data points."}
            continue

        valid_idx = series.index
        t = time_numeric[work.index.isin(valid_idx)]
        y = series.values

        slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
        direction = (
            "increasing" if slope > 0 and p_value < trend_pvalue_threshold
            else "decreasing" if slope < 0 and p_value < trend_pvalue_threshold
            else "flat"
        )

        pct_changes = series.pct_change().dropna()
        avg_growth = float(pct_changes.mean() * 100) if len(pct_changes) > 0 else 0.0

        n = len(series)
        split = int(n * 0.75)
        if split > 0 and split < n:
            early_growth = float(series.iloc[:split].pct_change().mean() * 100)
            late_growth = float(series.iloc[split:].pct_change().mean() * 100)
            acceleration = late_growth - early_growth
            acc_label = (
                "accelerating" if acceleration > 2
                else "decelerating" if acceleration < -2
                else "stable_growth"
            )
        else:
            acceleration, acc_label = 0.0, "insufficient_data"

        col_results[col] = {
            "trend_direction": direction,
            "trend_slope": round(float(slope), 6),
            "trend_r_squared": round(float(r_value ** 2), 4),
            "trend_pvalue": round(float(p_value), 6),
            "is_significant_trend": bool(p_value < trend_pvalue_threshold),
            "avg_period_growth_pct": round(float(avg_growth), 4),
            "acceleration_vs_early": round(float(acceleration), 4),
            "momentum_label": acc_label,
        }

    final_results = {
        "time_col": time_col,
        "freq_used": freq,
        "results": col_results,
        "trending_up": [c for c, v in col_results.items() if v.get("trend_direction") == "increasing"],
        "trending_down": [c for c, v in col_results.items() if v.get("trend_direction") == "decreasing"],
    }

    return Command(update={
        "analysis_results": to_serializable({"detect_time_trends_result": final_results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="detect_time_trends completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def detect_seasonality_hints(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    time_col: str,
    value_col: str,
) -> Command:
    """
    Detect potential seasonality patterns by analyzing monthly or quarterly averages.

    Args:
        time_col: Datetime column.
        value_col: Numeric metric to analyze.
    """
    df = load_df(state["clean_df_key"])
    if time_col not in df.columns or value_col not in df.columns:
        return Command(update={
            "analysis_results": to_serializable({"detect_seasonality_hints_result": {"error": "One or more required columns not found."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content="Error: required columns not found.", tool_call_id=tool_call_id)]
        })

    work = df[[time_col, value_col]].copy().dropna()
    try:
        work[time_col] = pd.to_datetime(work[time_col])
    except Exception:
        return Command(update={
            "analysis_results": to_serializable({"detect_seasonality_hints_result": {"error": f"Could not parse '{time_col}' as datetime."}}),
            "tool_priority_list_2": state["tool_priority_list_2"][1:],
            "messages": [ToolMessage(content=f"Error: could not parse '{time_col}'.", tool_call_id=tool_call_id)]
        })

    work["_month_"] = work[time_col].dt.month
    work["_quarter_"] = work[time_col].dt.quarter

    monthly = work.groupby("_month_")[value_col].mean().round(4)
    quarterly = work.groupby("_quarter_")[value_col].mean().round(4)
    month_cv = float(monthly.std() / monthly.mean()) if float(monthly.mean()) != 0 else 0.0

    interpretation = (
        "Strong seasonal pattern detected." if month_cv > 0.2
        else "Moderate seasonal variation." if month_cv > 0.08
        else "Weak or no seasonality detected."
    )

    results = {
        "value_col": value_col,
        "monthly_averages": {int(k): round(float(v), 4) for k, v in monthly.items()},
        "quarterly_averages": {int(k): round(float(v), 4) for k, v in quarterly.items()},
        "peak_month": int(monthly.idxmax()),
        "trough_month": int(monthly.idxmin()),
        "peak_quarter": int(quarterly.idxmax()),
        "trough_quarter": int(quarterly.idxmin()),
        "seasonality_strength_cv": round(month_cv, 4),
        "interpretation": interpretation,
    }

    return Command(update={
        "analysis_results": to_serializable({"detect_seasonality_hints_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="detect_seasonality_hints completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


# ============================================================
# MODULE 7 — CATEGORICAL DOMINANCE & DISTRIBUTION ANALYSIS
# ============================================================

@tool
def analyze_categorical_dominance(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    columns: list[str] | None = None,
    top_n: int = 10,
    dominance_threshold_pct: float = 50.0,
) -> Command:
    """
    Analyze distribution patterns across categorical columns.

    Args:
        columns: Categorical columns to analyze. Defaults to all object/category cols.
        top_n: Number of top categories to report.
        dominance_threshold_pct: If top category exceeds this %, flag as dominant.
    """
    df = load_df(state["clean_df_key"])
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    target_cols = [c for c in (columns or cat_cols) if c in cat_cols]

    results: dict[str, Any] = {}

    for col in target_cols:
        counts = df[col].value_counts(dropna=True)
        total = int(counts.sum())
        if total == 0:
            continue

        pcts = counts / total * 100
        top = [
            {"value": str(v), "count": int(counts[v]), "pct": round(float(pcts[v]), 2)}
            for v in counts.index[:top_n]
        ]

        probs = pcts / 100
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = float(np.log2(len(counts))) if len(counts) > 1 else 1.0
        normalized_entropy = round(entropy / max_entropy, 4)
        dominant = bool(float(pcts.iloc[0]) > dominance_threshold_pct)
        useful = bool(2 <= len(counts) <= 30 and not dominant)

        results[col] = {
            "total_unique": int(len(counts)),
            "top_categories": top,
            "dominant": dominant,
            "dominant_value": str(counts.index[0]),
            "dominant_pct": round(float(pcts.iloc[0]), 2),
            "normalized_entropy": normalized_entropy,
            "useful_for_segmentation": useful,
            "recommendation": (
                f"'{counts.index[0]}' dominates ({pcts.iloc[0]:.1f}%). May skew visualizations."
                if dominant else "Distribution is reasonably balanced for visualization."
            ),
        }

    return Command(update={
        "analysis_results": to_serializable({"analyze_categorical_dominance_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="analyze_categorical_dominance completed successfully",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def compute_categorical_numeric_relationships(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    top_n_cats: int = 20,
) -> Command:
    """
    Compute mean/median of numeric columns broken down by each categorical column.

    Args:
        cat_cols: Categorical columns. Defaults to all object/category cols.
        num_cols: Numeric columns to aggregate. Defaults to all numeric.
        top_n_cats: Max categories to include per column.
    """
    df = load_df(state["clean_df_key"])
    default_cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    default_nums = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in (cat_cols or default_cats) if c in default_cats]
    num_cols = [c for c in (num_cols or default_nums) if c in default_nums]

    results: dict[str, Any] = {}

    for cat_col in cat_cols:
        top_cats = df[cat_col].value_counts().head(top_n_cats).index
        sub = df[df[cat_col].isin(top_cats)]
        results[cat_col] = {}

        for num_col in num_cols:
            group = (
                sub.groupby(cat_col)[num_col]
                .agg(["mean", "median", "count"])
                .round(4)
                .sort_values("mean", ascending=False)
            )
            if len(group) < 2:
                continue

            group_list = [
                {
                    "category": str(idx),
                    "mean": round(float(row["mean"]), 4),
                    "median": round(float(row["median"]), 4),
                    "count": int(row["count"]),
                }
                for idx, row in group.iterrows()
            ]

            best_mean = float(group["mean"].max())
            worst_mean = float(group["mean"].min())
            spread = round(best_mean / worst_mean, 4) if worst_mean != 0 else None

            results[cat_col][num_col] = {
                "group_means": group_list,
                "best_category": str(group["mean"].idxmax()),
                "worst_category": str(group["mean"].idxmin()),
                "spread_ratio": spread,
            }

    return Command(update={
        "analysis_results": to_serializable({"compute_categorical_numeric_relationships_result": results}),
        "tool_priority_list_2": state["tool_priority_list_2"][1:],
        "messages": [ToolMessage(
            content="compute_categorical_numeric_relationships completed successfully",
            tool_call_id=tool_call_id,
        )]
    })