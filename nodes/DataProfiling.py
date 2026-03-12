from utils.state import AgentState
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


"""
================================================================================
  DATA PROFILING MODULE — Senior Data Analyst Grade
  For use in Data Analytics Agents on small-enterprise datasets
================================================================================
"""


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATASET OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────

def _get_dataset_overview(df: pd.DataFrame) -> dict:
    """High-level structural metadata about the dataset."""
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()

    duplicate_rows = df.duplicated().sum()

    return {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "total_cells": total_cells,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "duplicate_rows": int(duplicate_rows),
        "duplicate_rows_pct": round((duplicate_rows / df.shape[0]) * 100, 2),
        "total_missing_cells": int(total_missing),
        "overall_missing_pct": round((total_missing / total_cells) * 100, 2),
        "column_names": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — COMPLETENESS ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def _get_completeness_analysis(df: pd.DataFrame) -> dict:
    """Per-column null analysis + missing pattern signals."""
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df)) * 100

    # Masked nulls: whitespace-only or empty string
    masked_nulls = {}
    for col in df.select_dtypes(include="object").columns:
        masked = df[col].astype(str).str.strip().eq("").sum()
        if masked > 0:
            masked_nulls[col] = int(masked)

    per_column = {}
    for col in df.columns:
        pct = round(null_pcts[col], 2)
        per_column[col] = {
            "null_count": int(null_counts[col]),
            "null_pct": pct,
            "severity": (
                "critical" if pct > 80 else
                "high"     if pct > 50 else
                "moderate" if pct > 20 else
                "low"      if pct > 0  else
                "none"
            ),
        }

    # Rows with all nulls
    all_null_rows = int(df.isnull().all(axis=1).sum())

    # Missing pattern — are nulls correlated across columns?
    missing_corr_flags = []
    null_df = df.isnull().astype(int)
    if null_df.sum().sum() > 0:
        corr_matrix = null_df.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7:
                    missing_corr_flags.append({
                        "col_a": corr_matrix.columns[i],
                        "col_b": corr_matrix.columns[j],
                        "correlation": round(val, 3),
                        "signal": "Nulls tend to co-occur → likely MAR/structural"
                    })

    return {
        "per_column": per_column,
        "masked_null_columns": masked_nulls,
        "rows_entirely_null": all_null_rows,
        "missing_pattern_correlations": missing_corr_flags,
        "columns_above_50pct_missing": [c for c, v in per_column.items() if v["null_pct"] > 50],
        "columns_above_80pct_missing": [c for c, v in per_column.items() if v["null_pct"] > 80],
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — UNIVARIATE DISTRIBUTION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def _profile_numeric_column(series: pd.Series) -> dict:
    """Deep statistical profile for a numeric column."""
    clean = series.dropna()
    if len(clean) == 0:
        return {"error": "All values are null"}

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    skew_val = round(clean.skew(), 4)

    return {
        "count_non_null": int(len(clean)),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "mode": round(float(clean.mode().iloc[0]), 4) if not clean.mode().empty else None,
        "std_dev": round(float(clean.std()), 4),
        "variance": round(float(clean.var()), 4),
        "skewness": skew_val,
        "skewness_interpretation": (
            "highly right-skewed" if skew_val > 1 else
            "moderately right-skewed" if skew_val > 0.5 else
            "highly left-skewed" if skew_val < -1 else
            "moderately left-skewed" if skew_val < -0.5 else
            "approximately symmetric"
        ),
        "kurtosis": round(float(clean.kurtosis()), 4),
        "percentiles": {
            "p1":  round(float(clean.quantile(0.01)), 4),
            "p5":  round(float(clean.quantile(0.05)), 4),
            "p25": round(float(q1), 4),
            "p50": round(float(clean.quantile(0.50)), 4),
            "p75": round(float(q3), 4),
            "p95": round(float(clean.quantile(0.95)), 4),
            "p99": round(float(clean.quantile(0.99)), 4),
        },
        "iqr": round(float(iqr), 4),
        "zero_count": int((clean == 0).sum()),
        "negative_count": int((clean < 0).sum()),
        "positive_count": int((clean > 0).sum()),
        "iqr_outlier_count_lower": int((clean < (q1 - 1.5 * iqr)).sum()),
        "iqr_outlier_count_upper": int((clean > (q3 + 1.5 * iqr)).sum()),
        "zscore_outlier_count": int((np.abs(stats.zscore(clean)) > 3).sum()),
    }


def _profile_categorical_column(series: pd.Series, top_n: int = 10) -> dict:
    """Frequency and cardinality profile for categorical/object columns."""
    clean = series.dropna()
    total = len(clean)
    if total == 0:
        return {"error": "All values are null"}

    value_counts = clean.value_counts()
    top_values = value_counts.head(top_n)
    top_coverage = round((top_values.sum() / total) * 100, 2)

    rare_threshold = 0.01
    rare_categories = value_counts[value_counts / total < rare_threshold]

    return {
        "count_non_null": int(total),
        "unique_count": int(clean.nunique()),
        "cardinality_ratio": round(clean.nunique() / total, 4),
        "is_high_cardinality": clean.nunique() / total > 0.5,
        "is_binary": clean.nunique() == 2,
        "is_constant": clean.nunique() == 1,
        "top_values": {
            str(k): {"count": int(v), "pct": round((v / total) * 100, 2)}
            for k, v in top_values.items()
        },
        "top_n_coverage_pct": top_coverage,
        "rare_category_count": int(len(rare_categories)),
        "rare_categories": list(rare_categories.index[:10]),  # show up to 10
        "mode": str(value_counts.index[0]) if not value_counts.empty else None,
        "mode_frequency_pct": round((value_counts.iloc[0] / total) * 100, 2) if not value_counts.empty else None,
    }


def _profile_datetime_column(series: pd.Series) -> dict:
    """Temporal range and quality profile for datetime columns."""
    clean = pd.to_datetime(series, errors="coerce").dropna()
    if len(clean) == 0:
        return {"error": "No parseable datetime values"}

    now = pd.Timestamp(datetime.now())
    future_dates = int((clean > now).sum())
    epoch_zeros = int((clean == pd.Timestamp("1970-01-01")).sum())

    diffs = clean.sort_values().diff().dropna()
    common_interval = diffs.mode().iloc[0] if not diffs.mode().empty else None

    return {
        "count_non_null": int(len(clean)),
        "min_date": str(clean.min()),
        "max_date": str(clean.max()),
        "date_range_days": int((clean.max() - clean.min()).days),
        "future_dated_count": future_dates,
        "epoch_zero_count": epoch_zeros,
        "most_common_interval": str(common_interval) if common_interval else None,
        "inferred_granularity": (
            "sub-daily" if common_interval and common_interval < pd.Timedelta("1D") else
            "daily"     if common_interval and common_interval == pd.Timedelta("1D") else
            "weekly"    if common_interval and common_interval == pd.Timedelta("7D") else
            "monthly"   if common_interval and pd.Timedelta("28D") <= common_interval <= pd.Timedelta("31D") else
            "irregular/unknown"
        ),
    }


def _get_univariate_profiles(df: pd.DataFrame) -> dict:
    """Route each column to its appropriate profiler."""
    profiles = {}
    for col in df.columns:
        col_type = _infer_column_type(df[col])
        if col_type == "numeric":
            profiles[col] = {"inferred_type": "numeric", **_profile_numeric_column(df[col])}
        elif col_type == "datetime":
            profiles[col] = {"inferred_type": "datetime", **_profile_datetime_column(df[col])}
        else:
            profiles[col] = {"inferred_type": "categorical", **_profile_categorical_column(df[col])}
    return profiles


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — UNIQUENESS & KEY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def _get_key_analysis(df: pd.DataFrame) -> dict:
    """Identify candidate primary keys and semantic column roles."""
    n = len(df)
    candidate_keys = [col for col in df.columns if df[col].nunique() == n and df[col].notnull().all()]
    low_uniqueness = [col for col in df.columns if df[col].nunique() <= 5]

    # Composite key check on top uniqueness candidates
    composite_key_candidates = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(candidate_keys) == 0 and len(cat_cols) >= 2:
        for i in range(min(len(cat_cols), 5)):
            for j in range(i + 1, min(len(cat_cols), 5)):
                combo = df[[cat_cols[i], cat_cols[j]]].dropna()
                if combo.duplicated().sum() == 0 and len(combo) == n:
                    composite_key_candidates.append([cat_cols[i], cat_cols[j]])

    return {
        "candidate_primary_keys": candidate_keys,
        "low_uniqueness_columns": low_uniqueness,
        "composite_key_candidates": composite_key_candidates[:5],  # top 5
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DATA QUALITY FLAGS
# ──────────────────────────────────────────────────────────────────────────────

def _get_data_quality_flags(df: pd.DataFrame) -> dict:
    """Surface specific data quality issues per column."""
    issues = {}

    for col in df.columns:
        col_issues = []
        series = df[col]

        # Mixed type detection in object columns
        if series.dtype == object:
            sample = series.dropna().head(500)
            types_found = set(type(v).__name__ for v in sample)
            if len(types_found) > 1:
                col_issues.append(f"Mixed types detected: {types_found}")

            # Whitespace / empty strings
            whitespace_count = series.astype(str).str.strip().eq("").sum()
            if whitespace_count > 0:
                col_issues.append(f"Whitespace/empty strings: {whitespace_count} (masked nulls)")

            # Format inconsistency hints
            if sample.astype(str).str.contains(r"^\d{4}-\d{2}-\d{2}$", na=False).any():
                if not sample.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$", na=False).all():
                    col_issues.append("Inconsistent date-like format in string column")

        # Constant columns
        if series.nunique() == 1:
            col_issues.append("Constant column — zero variance, no analytical value")

        # Near-constant (std dev ≈ 0 for numeric)
        if pd.api.types.is_numeric_dtype(series):
            std = series.std()
            if std is not None and std < 1e-9 and series.nunique() > 1:
                col_issues.append("Near-constant numeric column — extremely low variance")

        if col_issues:
            issues[col] = col_issues

    return issues


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CORRELATION & RELATIONSHIP SIGNALS
# ──────────────────────────────────────────────────────────────────────────────

def _get_correlation_signals(df: pd.DataFrame) -> dict:
    """Pearson correlation matrix + high-correlation pair flags."""
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return {"note": "Fewer than 2 numeric columns — correlation skipped"}

    corr_matrix = numeric_df.corr(method="pearson")

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= 0.85:
                high_corr_pairs.append({
                    "col_a": corr_matrix.columns[i],
                    "col_b": corr_matrix.columns[j],
                    "pearson_r": round(val, 4),
                    "risk": "multicollinearity" if abs(val) >= 0.95 else "strong association",
                })

    return {
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "high_correlation_pairs": sorted(high_corr_pairs, key=lambda x: -abs(x["pearson_r"])),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — BUSINESS CONTEXT INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def _infer_column_type(series: pd.Series) -> str:
    """Infer numeric, datetime, or categorical."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    try:
        parsed = pd.to_datetime(series.dropna().head(100), errors="coerce")
        if parsed.notnull().mean() > 0.8:
            return "datetime"
    except Exception:
        pass
    return "categorical"


def _get_business_context(df: pd.DataFrame) -> dict:
    """Infer column roles, domain hints, and usability signals."""
    id_keywords = ["id", "key", "uuid", "code", "ref", "number", "no", "num"]
    date_keywords = ["date", "time", "timestamp", "created", "updated", "dt", "year", "month", "day"]
    target_keywords = ["target", "label", "output", "result", "churn", "fraud", "score", "status", "flag"]

    likely_ids, likely_dates, likely_targets, likely_unusable = [], [], [], []

    for col in df.columns:
        col_lower = col.lower()
        col_type = _infer_column_type(df[col])

        if any(kw in col_lower for kw in id_keywords):
            likely_ids.append(col)
        if any(kw in col_lower for kw in date_keywords) or col_type == "datetime":
            likely_dates.append(col)
        if any(kw in col_lower for kw in target_keywords):
            likely_targets.append(col)

        # Likely unusable: free text or constant
        if col_type == "categorical":
            n = len(df[col].dropna())
            if n > 0 and df[col].nunique() / n > 0.9 and n > 50:
                likely_unusable.append(f"{col} (high-cardinality text — possible free-text/ID)")
        if df[col].nunique() <= 1:
            likely_unusable.append(f"{col} (constant/empty — no analytical value)")

    # Naive domain inference from column name vocabulary
    all_cols_lower = " ".join(df.columns.str.lower())
    domain_hints = []
    if any(k in all_cols_lower for k in ["revenue", "sales", "price", "amount", "order", "product"]):
        domain_hints.append("Sales / E-commerce")
    if any(k in all_cols_lower for k in ["salary", "employee", "department", "hire", "leave", "payroll"]):
        domain_hints.append("HR / People Analytics")
    if any(k in all_cols_lower for k in ["balance", "transaction", "account", "credit", "debit", "loan"]):
        domain_hints.append("Finance / Banking")
    if any(k in all_cols_lower for k in ["patient", "diagnosis", "hospital", "drug", "symptom", "age", "bmi"]):
        domain_hints.append("Healthcare")
    if any(k in all_cols_lower for k in ["latitude", "longitude", "city", "region", "country", "zip"]):
        domain_hints.append("Geospatial / Location")
    if any(k in all_cols_lower for k in ["click", "session", "page", "user", "event", "visit", "impression"]):
        domain_hints.append("Web / Digital Analytics")
    if not domain_hints:
        domain_hints.append("Unknown — review column names manually")

    return {
        "likely_id_columns": likely_ids,
        "likely_date_columns": likely_dates,
        "likely_target_columns": likely_targets,
        "likely_unusable_columns": list(set(likely_unusable)),
        "inferred_domain_hints": domain_hints,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — AGENT PLANNING HINTS
# ──────────────────────────────────────────────────────────────────────────────

def _generate_agent_planning_hints(
    overview: dict,
    completeness: dict,
    univariate: dict,
    quality_flags: dict,
    correlations: dict,
    business_context: dict,
) -> list:
    """Translate profiling results into concrete agent action recommendations."""
    hints = []

    # Missing data actions
    high_missing = completeness["columns_above_50pct_missing"]
    critical_missing = completeness["columns_above_80pct_missing"]
    if critical_missing:
        hints.append({
            "priority": "HIGH",
            "category": "Data Cleaning",
            "action": f"Consider DROPPING columns with >80% missing: {critical_missing}",
        })
    if high_missing:
        hints.append({
            "priority": "HIGH",
            "category": "Imputation",
            "action": f"Columns with >50% missing need imputation strategy: {high_missing}",
        })

    # Recommend imputation per numeric col
    imputation_needed = []
    for col, profile in univariate.items():
        if profile.get("inferred_type") == "numeric":
            null_pct = completeness["per_column"][col]["null_pct"]
            if 0 < null_pct <= 50:
                skew = profile.get("skewness", 0)
                strategy = "median" if abs(skew) > 0.5 else "mean"
                imputation_needed.append(f"{col} → {strategy} imputation (skew={skew})")
        elif profile.get("inferred_type") == "categorical":
            null_pct = completeness["per_column"][col]["null_pct"]
            if 0 < null_pct <= 50:
                imputation_needed.append(f"{col} → mode or 'Unknown' imputation")
    if imputation_needed:
        hints.append({
            "priority": "MEDIUM",
            "category": "Imputation",
            "action": f"Imputation recommended: {imputation_needed}",
        })

    # Outlier handling
    outlier_cols = []
    for col, profile in univariate.items():
        if profile.get("inferred_type") == "numeric":
            total = profile.get("count_non_null", 1)
            iqr_out = profile.get("iqr_outlier_count_upper", 0) + profile.get("iqr_outlier_count_lower", 0)
            if total > 0 and iqr_out / total > 0.05:
                outlier_cols.append(f"{col} ({iqr_out} IQR outliers)")
    if outlier_cols:
        hints.append({
            "priority": "MEDIUM",
            "category": "Outlier Treatment",
            "action": f"Significant outliers detected — review for capping/removal: {outlier_cols}",
        })

    # Skewed columns — suggest log transform
    skewed_cols = [
        col for col, p in univariate.items()
        if p.get("inferred_type") == "numeric" and abs(p.get("skewness", 0)) > 1
    ]
    if skewed_cols:
        hints.append({
            "priority": "LOW",
            "category": "Feature Engineering",
            "action": f"Highly skewed columns may benefit from log/sqrt transform: {skewed_cols}",
        })

    # High cardinality categoricals — encoding advice
    high_card_cats = [
        col for col, p in univariate.items()
        if p.get("inferred_type") == "categorical" and p.get("is_high_cardinality")
    ]
    if high_card_cats:
        hints.append({
            "priority": "MEDIUM",
            "category": "Encoding",
            "action": f"High-cardinality categoricals — use target/frequency encoding, not one-hot: {high_card_cats}",
        })

    # Binary categoricals
    binary_cats = [
        col for col, p in univariate.items()
        if p.get("inferred_type") == "categorical" and p.get("is_binary")
    ]
    if binary_cats:
        hints.append({
            "priority": "LOW",
            "category": "Encoding",
            "action": f"Binary categoricals — label encode (0/1): {binary_cats}",
        })

    # Multicollinearity
    high_corr = correlations.get("high_correlation_pairs", [])
    if high_corr:
        pairs = [(p["col_a"], p["col_b"], p["pearson_r"]) for p in high_corr]
        hints.append({
            "priority": "MEDIUM",
            "category": "Feature Selection",
            "action": f"High correlation pairs detected — consider dropping one from each: {pairs}",
        })

    # Quality flag summary
    if quality_flags:
        hints.append({
            "priority": "HIGH",
            "category": "Data Quality",
            "action": f"Data quality issues in {len(quality_flags)} columns — review before analysis: {list(quality_flags.keys())}",
        })

    # Duplicate rows
    if overview["duplicate_rows"] > 0:
        hints.append({
            "priority": "HIGH",
            "category": "Deduplication",
            "action": f"{overview['duplicate_rows']} duplicate rows ({overview['duplicate_rows_pct']}%) — deduplicate before modeling",
        })

    # Likely target
    if business_context["likely_target_columns"]:
        hints.append({
            "priority": "INFO",
            "category": "Modeling",
            "action": f"Likely target/label columns detected: {business_context['likely_target_columns']}",
        })

    # Date columns → time series potential
    if business_context["likely_date_columns"]:
        hints.append({
            "priority": "INFO",
            "category": "Modeling",
            "action": f"Date columns present → time-series or cohort analysis may be viable: {business_context['likely_date_columns']}",
        })

    return hints


# ──────────────────────────────────────────────────────────────────────────────
# MASTER PROFILER — PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def profile_dataframe( state : AgentState ) -> AgentState:
    """
    Run a full senior-analyst-grade data profiling report on any DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile. Works for any mix of numeric,
        categorical, and datetime columns.
    top_n_categories : int
        How many top categories to surface per categorical column.

    Returns
    -------
    dict
        Structured profiling report with 8 sections + agent planning hints.
    """
    df=state["df"]

    # print("[ DATA PROFILER ] Starting profiling run...")

    # print("  ▸ Section 1 — Dataset Overview")
    overview = _get_dataset_overview(df)

    # print("  ▸ Section 2 — Completeness Analysis")
    completeness = _get_completeness_analysis(df)

    # print("  ▸ Section 3 — Univariate Profiles")
    univariate = _get_univariate_profiles(df)

    # print("  ▸ Section 4 — Key & Uniqueness Analysis")
    key_analysis = _get_key_analysis(df)

    # print("  ▸ Section 5 — Data Quality Flags")
    quality_flags = _get_data_quality_flags(df)

    # print("  ▸ Section 6 — Correlation Signals")
    correlations = _get_correlation_signals(df)

    # print("  ▸ Section 7 — Business Context Inference")
    business_context = _get_business_context(df)
    
    # print("  ▸ Section 8 — Agent Planning Hints")
    planning_hints = _generate_agent_planning_hints(
        overview, completeness, univariate,
        quality_flags, correlations, business_context
    )

    # print("[ DATA PROFILER ] Complete.\n")

    report= {
        "dataset_overview":      overview,
        "completeness_analysis": completeness,
        "univariate_profiles":   univariate,
        "key_analysis":          key_analysis,
        "data_quality_flags":    quality_flags,
        "correlation_signals":   correlations,
        "business_context":      business_context,
        "agent_planning_hints":  planning_hints,
    }
    
    return {'report': report}


# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: PRETTY PRINT UTILITY
# ──────────────────────────────────────────────────────────────────────────────

# def print_profile_summary(report: dict) -> None:
#     """Print a human-readable summary of the profiling report."""
#     ov = report["dataset_overview"]
    # print("=" * 70)
    # print("  DATA PROFILING SUMMARY REPORT")
    # print("=" * 70)
    # print(f"  Rows         : {ov['total_rows']:,}")
    # print(f"  Columns      : {ov['total_columns']}")
    # print(f"  Memory       : {ov['memory_usage_kb']} KB")
    # print(f"  Duplicates   : {ov['duplicate_rows']} rows ({ov['duplicate_rows_pct']}%)")
    # print(f"  Missing Cells: {ov['total_missing_cells']} ({ov['overall_missing_pct']}%)")
    # print()

    # print("── COLUMN SNAPSHOT ──────────────────────────────────────────────────")
#     for col, prof in report["univariate_profiles"].items():
#         null_info = report["completeness_analysis"]["per_column"][col]
#         t = prof.get("inferred_type", "?")
#         null_str = f"null={null_info['null_pct']}%"
#         if t == "numeric":
            # print(f"  [{t:11}] {col:<35} {null_str}  |  mean={prof.get('mean')}  std={prof.get('std_dev')}  skew={prof.get('skewness')}")
#         elif t == "categorical":
            # print(f"  [{t:11}] {col:<35} {null_str}  |  unique={prof.get('unique_count')}  mode='{prof.get('mode')}'")
#         elif t == "datetime":
            # print(f"  [{t:11}] {col:<35} {null_str}  |  range: {prof.get('min_date')} → {prof.get('max_date')}")

    # print()
    # # print("── AGENT PLANNING HINTS ─────────────────────────────────────────────")
#     for hint in report["agent_planning_hints"]:
        # print(f"  [{hint['priority']:6}] [{hint['category']}] {hint['action']}")
    # print("=" * 70)

