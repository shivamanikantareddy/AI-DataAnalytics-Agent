"""
AI Data Visualization Agent - Complete Tool Function Implementations
====================================================================
A production-grade visualization toolkit for use with AI data analysis agents.
Organized by visualization category with consistent API design.
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

from __future__ import annotations

import warnings
import logging
from pathlib import Path
from typing import Any, Optional, Union, Annotated
from langchain_core.tools import tool
from utils.state import AgentState
from langgraph.prebuilt import InjectedState

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt

from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.signal import periodogram

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import networkx as nx

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import geopandas as gpd
    import folium
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    DATASHADER_AVAILABLE = True
except ImportError:
    DATASHADER_AVAILABLE = False

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Visualization Normalizer----------------------------------


from plotly.tools import mpl_to_plotly
import matplotlib.figure
import plotly.graph_objects as go
import seaborn as sns

def convert_plotly(fig):
    return {
        "type": "plotly",
        "figure": fig.to_dict()
    }
    
def convert_matplotlib(fig):

    plotly_fig = mpl_to_plotly(fig)

    return {
        "type": "plotly",
        "figure": plotly_fig.to_dict()
    }
    
def convert_seaborn(grid):

    fig = grid.fig
    plotly_fig = mpl_to_plotly(fig)

    return {
        "type": "plotly",
        "figure": plotly_fig.to_dict()
    }


def normalize_chart(chart):

    if isinstance(chart, go.Figure):
        return convert_plotly(chart)

    if isinstance(chart, matplotlib.figure.Figure):
        return convert_matplotlib(chart)

    if isinstance(chart, sns.axisgrid.PairGrid):
        return convert_seaborn(chart)

    raise ValueError("Unsupported chart type")


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 14: DECISION SUPPORT HELPERS  (defined first – used everywhere)
# ─────────────────────────────────────────────────────────────────────────────

def remove_first(lst):
    return lst[1:]

def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return all numeric column names."""
    return list(df.select_dtypes(include="number").columns)

def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return all categorical / object / bool column names."""
    return list(df.select_dtypes(include=["object", "category", "bool"]).columns)

def validate_columns_exist(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if any column is missing from *df*."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 13: STYLING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

_THEME_DEFAULTS = {
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
}


def apply_standard_theme() -> None:
    """Apply the project-wide matplotlib theme."""
    plt.rcParams.update(_THEME_DEFAULTS)
    sns.set_theme(style="whitegrid", font_scale=1.05)


def apply_color_palette(palette_name: str = "viridis") -> list:
    """Set seaborn / matplotlib colour palette and return the colour list."""
    sns.set_palette(palette_name)
    return sns.color_palette(palette_name)


def format_axis_labels(ax: matplotlib.axes.Axes, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Apply formatted labels to a matplotlib Axes object."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=9)


def apply_grid_style(ax: matplotlib.axes.Axes) -> None:
    """Apply consistent grid styling."""
    ax.grid(True, linestyle="--", linewidth=0.6, color="#E5E5E5", alpha=0.9)
    ax.set_axisbelow(True)


def annotate_significant_points(
    ax: matplotlib.axes.Axes,
    x_vals: list,
    y_vals: list,
    labels: list,
    color: str = "crimson",
) -> None:
    """Annotate specific data points on a matplotlib Axes."""
    for xv, yv, lbl in zip(x_vals, y_vals, labels):
        ax.annotate(
            lbl,
            xy=(xv, yv),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )


def _save(fig: Any, save_path: Optional[str], dpi: int = 150) -> None:
    """Save a matplotlib or plotly figure if *save_path* is provided."""
    if save_path is None:
        return
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(fig, plt.Figure):
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved matplotlib figure → %s", save_path)
    
    elif hasattr(fig, "write_image"):
        fig.write_html(save_path)
        logger.info("Saved plotly figure → %s", save_path)
    
    elif hasattr(fig, "save"):
        fig.save(save_path)
        logger.info("Saved altair chart → %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 1: UNIVARIATE VISUALIZATION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_histogram(
    state: Annotated[ AgentState, InjectedState],
    column: str,
    bins: int = 30,
    kde: bool = True,
    title: Optional[str] = None,
    color: str = "#4C72B0",
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram with optional KDE overlay for a numeric column.

    Parameters
    ----------
    column : numeric column to plot
    bins : number of histogram bins
    kde : overlay kernel density estimate
    title : chart title (auto-generated if None)
    color : bar fill colour

    Returns
    -------
    matplotlib Figure
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [column])
    apply_standard_theme()
    series = df[column].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(series, bins=bins, kde=kde, color=color, ax=ax, edgecolor="white", linewidth=0.4)
    format_axis_labels(ax, xlabel=column, ylabel="Frequency", title=title or f"Distribution of {column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


@tool
def create_kde_plot(
    state: Annotated[ AgentState, InjectedState],
    column: str,
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Kernel density estimate plot, optionally grouped by a categorical column.

    Parameters
    ----------
    column : numeric column
    group_by : optional categorical column for multi-group KDE
    title : chart title

    Returns
    -------
    matplotlib Figure
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [column] + ([group_by] if group_by else []))
    apply_standard_theme()
    fig, ax = plt.subplots(figsize=(9, 5))
    if group_by:
        for grp, sub in df.groupby(group_by):
            sns.kdeplot(sub[column].dropna(), ax=ax, label=str(grp), fill=True, alpha=0.3)
        ax.legend(title=group_by)
    else:
        sns.kdeplot(df[column].dropna(), ax=ax, fill=True, alpha=0.4, color="#4C72B0")
    format_axis_labels(ax, xlabel=column, ylabel="Density", title=title or f"Density of {column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


@tool
def create_box_plot(
    state: Annotated[ AgentState, InjectedState],
    column: str,
    title: Optional[str] = None,
    color: str = "#55A868",
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Box plot for a single numeric column."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [column])
    apply_standard_theme()
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(y=df[column].dropna(), ax=ax, color=color, width=0.4, linewidth=1.2)
    format_axis_labels(ax, ylabel=column, title=title or f"Box Plot – {column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


@tool
def create_violin_plot(
    state: Annotated[ AgentState, InjectedState],
    column: str,
    title: Optional[str] = None,
    color: str = "#C44E52",
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Violin plot for a single numeric column."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [column])
    apply_standard_theme()
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.violinplot(y=df[column].dropna(), ax=ax, color=color, inner="quartile", linewidth=1)
    format_axis_labels(ax, ylabel=column, title=title or f"Violin Plot – {column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


@tool
def create_frequency_bar_chart(
    state: Annotated[ AgentState, InjectedState],
    column: str,
    top_n: int = 20,
    title: Optional[str] = None,
    color: str = "#4C72B0",
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of value counts for a categorical column."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [column])
    apply_standard_theme()
    counts = df[column].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.4)))
    counts[::-1].plot(kind="barh", ax=ax, color=color, edgecolor="white")
    format_axis_labels(ax, xlabel="Count", ylabel=column, title=title or f"Top {top_n} – {column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_pie_chart(
    state: Annotated[ AgentState, InjectedState],
    column: str,
    top_n: int = 8,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Pie chart for proportional breakdown of a categorical column."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [column])
    apply_standard_theme()
    counts = df[column].value_counts().head(top_n)
    if len(df[column].value_counts()) > top_n:
        counts["Other"] = df[column].value_counts().iloc[top_n:].sum()
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("tab10", len(counts)),
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title(title or f"Proportion – {column}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 2: BIVARIATE VISUALIZATION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_scatter_plot(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """
    Interactive Plotly scatter plot for two numeric columns.

    Parameters
    ----------
    x_column : x-axis numeric column
    y_column : y-axis numeric column
    color_column : optional column to encode colour
    size_column : optional numeric column to encode point size
    title : chart title
    save_path : optional HTML/PNG export path

    Returns
    -------
    plotly Figure
    """
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    cols = [x_column, y_column] + [c for c in (color_column, size_column) if c]
    validate_columns_exist(df, cols)
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        size=size_column,
        title=title or f"{y_column} vs {x_column}",
        template="plotly_white",
        opacity=0.75,
        hover_data=df.columns.tolist(),
    )
    fig.update_layout(title_font_size=16, font_size=12)
    _save(fig, save_path)
    # return fig


@tool
def create_regression_plot(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter with OLS regression line and 95% CI band."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column])
    apply_standard_theme()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.regplot(data=df, x=x_column, y=y_column, ax=ax, scatter_kws={"alpha": 0.5, "s": 25}, line_kws={"color": "crimson"})
    slope, intercept, r, p, _ = stats.linregress(df[x_column].dropna(), df[y_column].dropna())
    ax.annotate(f"r={r:.3f}  p={p:.3e}", xy=(0.05, 0.93), xycoords="axes fraction", fontsize=9, color="crimson")
    format_axis_labels(ax, xlabel=x_column, ylabel=y_column, title=title or f"Regression: {y_column} ~ {x_column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_grouped_bar_chart(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    group_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart comparing a numeric metric across categories."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column, group_column])
    apply_standard_theme()
    pivot = df.groupby([x_column, group_column])[y_column].mean().unstack()
    ax = pivot.plot(kind="bar", figsize=(11, 6), edgecolor="white", width=0.75)
    format_axis_labels(ax, xlabel=x_column, ylabel=f"Mean {y_column}", title=title or f"{y_column} by {x_column} & {group_column}")
    ax.legend(title=group_column, bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_grid_style(ax)
    plt.xticks(rotation=30, ha="right")
    fig = ax.get_figure()
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_box_plot_by_category(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Box plot of a numeric column grouped by a categorical column."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column])
    apply_standard_theme()
    order = df.groupby(x_column)[y_column].median().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9), 6))
    sns.boxplot(data=df, x=x_column, y=y_column, order=order, ax=ax, palette="Set2")
    format_axis_labels(ax, xlabel=x_column, ylabel=y_column, title=title or f"{y_column} by {x_column}")
    apply_grid_style(ax)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_violin_plot_by_category(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Violin plot of a numeric column grouped by a categorical column."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column])
    apply_standard_theme()
    order = df.groupby(x_column)[y_column].median().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9), 6))
    sns.violinplot(data=df, x=x_column, y=y_column, order=order, ax=ax, palette="Set3", inner="quartile")
    format_axis_labels(ax, xlabel=x_column, ylabel=y_column, title=title or f"{y_column} by {x_column} (Violin)")
    apply_grid_style(ax)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_categorical_comparison_chart(
    state: Annotated[ AgentState, InjectedState],
    col1: str,
    col2: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of co-occurrence counts for two categorical columns."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [col1, col2])
    apply_standard_theme()
    ct = pd.crosstab(df[col1], df[col2])
    fig, ax = plt.subplots(figsize=(max(7, ct.shape[1] * 1.1), max(5, ct.shape[0] * 0.6)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.4, linecolor="white")
    format_axis_labels(ax, xlabel=col2, ylabel=col1, title=title or f"Co-occurrence: {col1} × {col2}")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 3: MULTIVARIATE VISUALIZATION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_pair_plot(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> sns.PairGrid:
    """
    Seaborn pairplot for multivariate numeric relationships.

    Parameters
    ----------
    columns : numeric columns to include (defaults to all numeric)
    hue : optional categorical column for colour encoding
    title : suptitle
    save_path : optional file path

    Returns
    -------
    seaborn PairGrid
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    apply_standard_theme()
    if columns is None:
        columns = get_numeric_columns(df)
    subset = df[columns + ([hue] if hue else [])].dropna()
    g = sns.pairplot(subset, hue=hue, plot_kws={"alpha": 0.5, "s": 15}, diag_kind="kde")
    if title:
        g.figure.suptitle(title, y=1.01, fontsize=14, fontweight="bold")
    g.figure.tight_layout()
    _save(g.figure, save_path)
    return g


@tool
def create_correlation_heatmap(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    method: str = "pearson",
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Annotated correlation matrix heatmap.

    Parameters
    ----------
    columns : numeric columns (defaults to all)
    method : 'pearson' | 'spearman' | 'kendall'
    title : chart title
    save_path : optional file path

    Returns
    -------
    matplotlib Figure
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    apply_standard_theme()
    if columns is None:
        columns = get_numeric_columns(df)
    validate_columns_exist(df, columns)
    corr = df[columns].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 0.9), max(7, len(columns) * 0.8)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 8}, vmin=-1, vmax=1,
    )
    format_axis_labels(ax, title=title or f"Correlation Matrix ({method.capitalize()})")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_bubble_chart(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    size_column: str,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Interactive Plotly bubble chart."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    cols = [x_column, y_column, size_column] + ([color_column] if color_column else [])
    validate_columns_exist(df, cols)
    fig = px.scatter(
        df, x=x_column, y=y_column,
        size=size_column, color=color_column,
        title=title or f"Bubble Chart: {x_column} vs {y_column}",
        template="plotly_white", hover_data=df.columns.tolist(), opacity=0.7,
    )
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_grouped_scatter_plot(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    group_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Matplotlib scatter plot with colour-coded groups."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column, group_column])
    apply_standard_theme()
    groups = df[group_column].unique()
    palette = sns.color_palette("tab10", len(groups))
    fig, ax = plt.subplots(figsize=(10, 6))
    for grp, col in zip(groups, palette):
        sub = df[df[group_column] == grp]
        ax.scatter(sub[x_column], sub[y_column], label=str(grp), color=col, alpha=0.6, s=25)
    ax.legend(title=group_column, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    format_axis_labels(ax, xlabel=x_column, ylabel=y_column, title=title or f"{y_column} vs {x_column} by {group_column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_stacked_bar_chart(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    stack_column: str,
    normalize: bool = False,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Stacked (or 100%-normalised) bar chart."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column, stack_column])
    apply_standard_theme()
    pivot = df.groupby([x_column, stack_column])[y_column].sum().unstack(fill_value=0)
    if normalize:
        pivot = pivot.div(pivot.sum(axis=1), axis=0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="tab20", edgecolor="white", width=0.75)
    ylabel = "Proportion" if normalize else f"Total {y_column}"
    format_axis_labels(ax, xlabel=x_column, ylabel=ylabel, title=title or f"Stacked Bar – {y_column}")
    ax.legend(title=stack_column, bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_grid_style(ax)
    plt.xticks(rotation=30, ha="right")
    fig = ax.get_figure()
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_cluster_heatmap(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Hierarchically-clustered heatmap using seaborn clustermap."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    apply_standard_theme()
    if columns is None:
        columns = get_numeric_columns(df)
    validate_columns_exist(df, columns)
    data = df[columns].dropna()
    g = sns.clustermap(
        data.T, cmap="RdBu_r", center=0, standard_scale=1,
        figsize=(max(10, len(data.columns) * 0.15), max(7, len(columns) * 0.5)),
        linewidths=0.01, yticklabels=True,
    )
    g.fig.suptitle(title or "Cluster Heatmap", fontsize=14, fontweight="bold", y=1.01)
    _save(g.fig, save_path)
    return g.fig


@tool
def create_parallel_coordinates_plot(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Interactive parallel coordinates plot via Plotly."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    if columns is None:
        columns = get_numeric_columns(df)
    validate_columns_exist(df, columns + ([color_column] if color_column else []))
    dims = [dict(range=[df[c].min(), df[c].max()], label=c, values=df[c]) for c in columns]
    if color_column and color_column in get_numeric_columns(df):
        color_vals = df[color_column]
        colorscale = "Viridis"
    else:
        color_vals = np.zeros(len(df))
        colorscale = "Blues"
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=color_vals, colorscale=colorscale, showscale=True),
        dimensions=dims,
    ))
    fig.update_layout(title=title or "Parallel Coordinates", template="plotly_white", title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_radar_chart(
    state: Annotated[ AgentState, InjectedState],
    category_column: str,
    value_columns: list[str],
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Spider/radar chart comparing multiple metrics across categories."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [category_column] + value_columns)
    agg = df.groupby(category_column)[value_columns].mean()
    fig = go.Figure()
    for cat, row in agg.iterrows():
        vals = list(row.values) + [row.values[0]]
        cats = value_columns + [value_columns[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name=str(cat)))
    fig.update_layout(
        title=title or f"Radar Chart – {category_column}",
        template="plotly_white", title_font_size=16,
        polar=dict(radialaxis=dict(visible=True)),
    )
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 4: TIME SERIES VISUALIZATION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_time_series_line_chart(
    state: Annotated[ AgentState, InjectedState],
    date_column: str,
    value_column: str,
    group_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line chart of a numeric metric over time.

    Parameters
    ----------
    date_column : datetime column
    value_column : numeric column to plot
    group_column : optional categorical column for multi-line chart
    title : chart title
    save_path : optional file path

    Returns
    -------
    matplotlib Figure
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [date_column, value_column])
    apply_standard_theme()
    dfc = df.copy()
    dfc[date_column] = pd.to_datetime(dfc[date_column])
    dfc = dfc.sort_values(date_column)
    fig, ax = plt.subplots(figsize=(13, 5))
    if group_column:
        validate_columns_exist(df, [group_column])
        for grp, sub in dfc.groupby(group_column):
            ax.plot(sub[date_column], sub[value_column], label=str(grp), linewidth=1.5)
        ax.legend(title=group_column, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.plot(dfc[date_column], dfc[value_column], color="#4C72B0", linewidth=1.5)
    format_axis_labels(ax, xlabel=date_column, ylabel=value_column, title=title or f"{value_column} Over Time")
    apply_grid_style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_moving_average_chart(
    state: Annotated[ AgentState, InjectedState],
    date_column: str,
    value_column: str,
    windows: list[int] = (7, 30),
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Time series with configurable moving-average overlays."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [date_column, value_column])
    apply_standard_theme()
    dfc = df.copy()
    dfc[date_column] = pd.to_datetime(dfc[date_column])
    dfc = dfc.sort_values(date_column)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dfc[date_column], dfc[value_column], color="steelblue", alpha=0.4, linewidth=1, label="Raw")
    palette = ["crimson", "darkorange", "green", "purple"]
    for w, c in zip(windows, palette):
        ma = dfc[value_column].rolling(w).mean()
        ax.plot(dfc[date_column], ma, color=c, linewidth=1.8, label=f"MA-{w}")
    ax.legend()
    format_axis_labels(ax, xlabel=date_column, ylabel=value_column, title=title or f"{value_column} – Moving Averages")
    apply_grid_style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_seasonal_decomposition_plot(
    state: Annotated[ AgentState, InjectedState],
    date_column: str,
    value_column: str,
    period: int = 12,
    model: str = "additive",
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Seasonal decomposition (trend + seasonal + residual) chart."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [date_column, value_column])
    apply_standard_theme()
    dfc = df[[date_column, value_column]].copy()
    dfc[date_column] = pd.to_datetime(dfc[date_column])
    dfc = dfc.set_index(date_column).sort_index()
    ts = dfc[value_column].dropna()
    result = seasonal_decompose(ts, model=model, period=period, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for ax, data, lbl in zip(axes, [ts, result.trend, result.seasonal, result.resid],
                              ["Observed", "Trend", "Seasonal", "Residual"]):
        ax.plot(data, linewidth=1.2)
        ax.set_ylabel(lbl, fontsize=9)
        apply_grid_style(ax)
    axes[0].set_title(title or f"Seasonal Decomposition – {value_column}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_time_series_comparison_chart(
    state: Annotated[ AgentState, InjectedState],
    date_column: str,
    value_columns: list[str],
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-line chart comparing several time series on the same axes."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [date_column] + value_columns)
    apply_standard_theme()
    dfc = df.copy()
    dfc[date_column] = pd.to_datetime(dfc[date_column])
    dfc = dfc.sort_values(date_column)
    fig, ax = plt.subplots(figsize=(13, 5))
    palette = sns.color_palette("tab10", len(value_columns))
    for col, clr in zip(value_columns, palette):
        ax.plot(dfc[date_column], dfc[col], label=col, color=clr, linewidth=1.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    format_axis_labels(ax, xlabel=date_column, ylabel="Value", title=title or "Time Series Comparison")
    apply_grid_style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_time_series_area_chart(
    state: Annotated[ AgentState, InjectedState],
    date_column: str,
    value_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Area chart for a single time series."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [date_column, value_column])
    apply_standard_theme()
    dfc = df.copy()
    dfc[date_column] = pd.to_datetime(dfc[date_column])
    dfc = dfc.sort_values(date_column)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(dfc[date_column], dfc[value_column], alpha=0.4, color="steelblue")
    ax.plot(dfc[date_column], dfc[value_column], color="steelblue", linewidth=1.5)
    format_axis_labels(ax, xlabel=date_column, ylabel=value_column, title=title or f"{value_column} – Area Chart")
    apply_grid_style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 5: CORRELATION AND STATISTICAL VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_correlation_matrix_heatmap(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    method: str = "pearson",
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Full (non-triangular) correlation matrix with significance markers."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    # Delegates to the existing function for DRY compliance.
    return create_correlation_heatmap(df, columns=columns, method=method, title=title, save_path=save_path)


@tool
def create_regression_analysis_visualization(
    state: Annotated[ AgentState, InjectedState],
    x_columns: list[str],
    y_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """OLS coefficient plot with 95% CIs from statsmodels."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, x_columns + [y_column])
    apply_standard_theme()
    dfc = df[x_columns + [y_column]].dropna()
    X = sm.add_constant(dfc[x_columns])
    model = sm.OLS(dfc[y_column], X).fit()
    params = model.params.drop("const")
    cis = model.conf_int().drop("const")
    fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.55)))
    y_pos = range(len(params))
    ax.barh(y_pos, params.values, xerr=[params.values - cis[0].values, cis[1].values - params.values],
            color=["#C44E52" if p < 0 else "#4C72B0" for p in params.values],
            align="center", alpha=0.8, capsize=4)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(params.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    format_axis_labels(ax, xlabel="Coefficient", title=title or f"OLS Coefficients – {y_column}")
    ax.annotate(f"R² = {model.rsquared:.3f}  adj-R² = {model.rsquared_adj:.3f}", xy=(0.98, 0.02),
                xycoords="axes fraction", ha="right", fontsize=9)
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_residual_plot(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Residual vs fitted values plot for a simple OLS model."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column])
    apply_standard_theme()
    sub = df[[x_column, y_column]].dropna()
    X = sm.add_constant(sub[x_column])
    model = sm.OLS(sub[y_column], X).fit()
    fitted = model.fittedvalues
    residuals = model.resid
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(fitted, residuals, alpha=0.5, s=20, color="steelblue")
    axes[0].axhline(0, color="red", linewidth=1, linestyle="--")
    format_axis_labels(axes[0], xlabel="Fitted Values", ylabel="Residuals", title="Residuals vs Fitted")
    apply_grid_style(axes[0])
    sm.qqplot(residuals, line="s", ax=axes[1], alpha=0.5)
    axes[1].set_title("Normal Q–Q Plot", fontsize=12, fontweight="bold")
    apply_grid_style(axes[1])
    fig.suptitle(title or f"Residual Diagnostics – {y_column} ~ {x_column}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_distribution_comparison_chart(
    state: Annotated[ AgentState, InjectedState],
    column: str,
    group_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlapping KDE distributions for each group."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [column, group_column])
    apply_standard_theme()
    groups = df[group_column].unique()
    palette = sns.color_palette("tab10", len(groups))
    fig, ax = plt.subplots(figsize=(10, 5))
    for grp, clr in zip(groups, palette):
        sub = df[df[group_column] == grp][column].dropna()
        sns.kdeplot(sub, ax=ax, label=str(grp), color=clr, fill=True, alpha=0.2)
    ax.legend(title=group_column)
    format_axis_labels(ax, xlabel=column, ylabel="Density", title=title or f"Distribution of {column} by {group_column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_statistical_significance_visualization(
    state: Annotated[ AgentState, InjectedState],
    column: str,
    group_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of group means with SEM error bars and ANOVA p-value annotation."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [column, group_column])
    apply_standard_theme()
    grouped = df.groupby(group_column)[column]
    means = grouped.mean()
    sems = grouped.sem()
    groups = [df[df[group_column] == g][column].dropna().values for g in means.index]
    f_stat, p_val = stats.f_oneway(*groups)
    fig, ax = plt.subplots(figsize=(max(7, len(means) * 0.9), 6))
    bars = ax.bar(means.index, means.values, yerr=sems.values, capsize=4,
                  color=sns.color_palette("tab10", len(means)), edgecolor="white", alpha=0.85)
    ax.annotate(f"ANOVA: F={f_stat:.2f}, p={p_val:.4f}", xy=(0.5, 0.97), xycoords="axes fraction",
                ha="center", fontsize=10, color="crimson" if p_val < 0.05 else "gray")
    format_axis_labels(ax, xlabel=group_column, ylabel=f"Mean {column}", title=title or f"Group Means – {column}")
    apply_grid_style(ax)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 6: GEOSPATIAL VISUALIZATION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_choropleth_map(
    state: Annotated[ AgentState, InjectedState],
    location_column: str,
    value_column: str,
    location_mode: str = "country names",
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """
    Choropleth world map via Plotly.

    Parameters
    ----------
    location_column : column with country/ISO names or codes
    value_column : numeric column to encode colour
    location_mode : 'country names' | 'ISO-3' | 'USA-states'
    title : chart title
    save_path : optional file path

    Returns
    -------
    plotly Figure
    """
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [location_column, value_column])
    fig = px.choropleth(
        df, locations=location_column, color=value_column,
        locationmode=location_mode, color_continuous_scale="Viridis",
        title=title or f"Choropleth – {value_column}",
        template="plotly_white",
    )
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_geospatial_scatter_map(
    state: Annotated[ AgentState, InjectedState],
    lat_column: str,
    lon_column: str,
    value_column: Optional[str] = None,
    hover_columns: Optional[list[str]] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Scatter map using lat/lon coordinates."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    cols = [lat_column, lon_column] + ([value_column] if value_column else [])
    validate_columns_exist(df, cols)
    fig = px.scatter_geo(
        df, lat=lat_column, lon=lon_column, color=value_column,
        hover_data=hover_columns or df.columns.tolist(),
        title=title or "Geospatial Scatter Map",
        template="plotly_white", color_continuous_scale="Plasma",
    )
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_location_density_heatmap(
    state: Annotated[ AgentState, InjectedState],
    lat_column: str,
    lon_column: str,
    value_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Density heatmap on a map using Plotly."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [lat_column, lon_column])
    fig = px.density_mapbox(
        df, lat=lat_column, lon=lon_column, z=value_column,
        radius=10, center=dict(lat=df[lat_column].mean(), lon=df[lon_column].mean()),
        zoom=3, mapbox_style="open-street-map",
        title=title or "Location Density Heatmap",
        color_continuous_scale="Hot",
    )
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_regional_comparison_map(
    state: Annotated[ AgentState, InjectedState],
    location_column: str,
    value_column: str,
    scope: str = "world",
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Choropleth scoped to a specific region (e.g., 'usa', 'europe')."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [location_column, value_column])
    fig = px.choropleth(
        df, locations=location_column, color=value_column,
        locationmode="country names", scope=scope,
        color_continuous_scale="RdYlGn",
        title=title or f"Regional Comparison – {value_column}",
        template="plotly_white",
    )
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 7: NETWORK GRAPH VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_nx_graph(edges_df: pd.DataFrame, source_col: str, target_col: str, weight_col: Optional[str]) -> nx.Graph:
    G = nx.from_pandas_edgelist(edges_df, source=source_col, target=target_col,
                                 edge_attr=weight_col, create_using=nx.Graph())
    return G


@tool
def create_node_link_graph(
    state: Annotated[ AgentState, InjectedState],
    source_column: str,
    target_column: str,
    weight_column: Optional[str] = None,
    layout: str = "spring",
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """
    NetworkX node-link graph.

    Parameters
    ----------
    source_column : source node column
    target_column : target node column
    weight_column : optional edge weight column
    layout : 'spring' | 'circular' | 'kamada_kawai'
    title : chart title
    save_path : optional file path

    Returns
    -------
    matplotlib Figure
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [source_column, target_column])
    apply_standard_theme()
    G = _build_nx_graph(df, source_column, target_column, weight_column)
    layouts = {"spring": nx.spring_layout, "circular": nx.circular_layout, "kamada_kawai": nx.kamada_kawai_layout}
    pos = layouts.get(layout, nx.spring_layout)(G, seed=42)
    degree = dict(G.degree())
    node_sizes = [300 + degree[n] * 80 for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx(G, pos=pos, ax=ax, node_size=node_sizes, node_color="steelblue",
                     edge_color="gray", font_size=7, alpha=0.85, with_labels=True)
    ax.set_title(title or "Node-Link Graph", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_dependency_graph(
    state: Annotated[ AgentState, InjectedState],
    source_column: str,
    target_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Interactive directed dependency graph via Plotly."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [source_column, target_column])
    G = nx.from_pandas_edgelist(df, source=source_column, target=target_column,
                                 create_using=nx.DiGraph())
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.8, color="#888"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                              text=list(G.nodes()), textposition="top center",
                              marker=dict(size=10, color="steelblue", line_width=1)))
    fig.update_layout(title=title or "Dependency Graph", template="plotly_white",
                       showlegend=False, title_font_size=16,
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    _save(fig, save_path)
    # return fig


@tool
def create_relationship_network_graph(
    state: Annotated[ AgentState, InjectedState],
    source_column: str,
    target_column: str,
    weight_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Weighted relationship network with edge thickness encoding."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [source_column, target_column])
    G = _build_nx_graph(df, source_column, target_column, weight_column)
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()
    max_w = max((d.get(weight_column or "weight", 1) for _, _, d in G.edges(data=True)), default=1)
    for u, v, data in G.edges(data=True):
        w = data.get(weight_column or "weight", 1)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                                  line=dict(width=max(0.5, w / max_w * 5), color="gray"),
                                  hoverinfo="none"))
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    deg = [G.degree(n) for n in G.nodes()]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                              text=list(G.nodes()), textposition="top center",
                              marker=dict(size=[6 + d * 3 for d in deg], color="steelblue",
                                          colorscale="Blues", line_width=1)))
    fig.update_layout(title=title or "Relationship Network", template="plotly_white",
                       showlegend=False, title_font_size=16,
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 8: DIMENSIONALITY REDUCTION VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_numeric_matrix(df : pd.DataFrame, columns: Optional[list[str]]) -> tuple[np.ndarray, list[str]]:
    if columns is None:
        columns = get_numeric_columns(df)
    validate_columns_exist(df, columns)
    X = df[columns].dropna().values
    scaler = StandardScaler()
    return scaler.fit_transform(X), columns


@tool
def create_pca_visualization(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    n_components: int = 2,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """
    Interactive PCA projection scatter plot.

    Parameters
    ----------
    columns : numeric features to reduce (defaults to all numeric)
    n_components : 2 or 3
    color_column : optional categorical column for colour
    title : chart title
    save_path : optional file path

    Returns
    -------
    plotly Figure
    """
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    X_scaled, _ = _prepare_numeric_matrix(df, columns)
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    idx = df[get_numeric_columns(df)[0] if not columns else columns[0]].dropna().index
    plot_df = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])], index=idx)
    if color_column and color_column in df.columns:
        plot_df[color_column] = df.loc[idx, color_column].values
    if n_components == 3 and coords.shape[1] == 3:
        fig = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color=color_column,
                             title=title or "PCA – 3D Projection", template="plotly_white")
    else:
        fig = px.scatter(plot_df, x="PC1", y="PC2", color=color_column,
                          title=title or f"PCA – PC1 ({var[0]:.1%}) vs PC2 ({var[1]:.1%})",
                          template="plotly_white", opacity=0.7)
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_tsne_plot(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    perplexity: float = 30.0,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """t-SNE 2-D projection scatter plot."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    X_scaled, feat_cols = _prepare_numeric_matrix(df, columns)
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(X_scaled) - 1), random_state=42, n_iter=1000)
    coords = tsne.fit_transform(X_scaled)
    idx = df[feat_cols[0]].dropna().index
    plot_df = pd.DataFrame({"tSNE-1": coords[:, 0], "tSNE-2": coords[:, 1]}, index=idx)
    if color_column and color_column in df.columns:
        plot_df[color_column] = df.loc[idx, color_column].values
    fig = px.scatter(plot_df, x="tSNE-1", y="tSNE-2", color=color_column,
                      title=title or "t-SNE Projection", template="plotly_white", opacity=0.7)
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_umap_plot(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """UMAP 2-D projection scatter plot (requires the `umap-learn` package)."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is not installed. Run: pip install umap-learn")
    X_scaled, feat_cols = _prepare_numeric_matrix(df, columns)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(X_scaled)
    idx = df[feat_cols[0]].dropna().index
    plot_df = pd.DataFrame({"UMAP-1": coords[:, 0], "UMAP-2": coords[:, 1]}, index=idx)
    if color_column and color_column in df.columns:
        plot_df[color_column] = df.loc[idx, color_column].values
    fig = px.scatter(plot_df, x="UMAP-1", y="UMAP-2", color=color_column,
                      title=title or "UMAP Projection", template="plotly_white", opacity=0.7)
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 9: HIERARCHICAL DATA VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_treemap(
    state: Annotated[ AgentState, InjectedState],
    path_columns: list[str],
    value_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plotly treemap for hierarchical category proportions.

    Parameters
    ----------
    path_columns : ordered list of categorical columns (hierarchy levels)
    value_column : numeric column for rectangle size
    title : chart title
    save_path : optional file path

    Returns
    -------
    plotly Figure
    """
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, path_columns + [value_column])
    fig = px.treemap(df, path=path_columns, values=value_column,
                      title=title or "Treemap", color=value_column,
                      color_continuous_scale="RdBu", template="plotly_white")
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_sunburst_chart(
    state: Annotated[ AgentState, InjectedState],
    path_columns: list[str],
    value_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Plotly sunburst chart for hierarchical proportions."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, path_columns + [value_column])
    fig = px.sunburst(df, path=path_columns, values=value_column,
                       title=title or "Sunburst Chart", color=value_column,
                       color_continuous_scale="Sunset", template="plotly_white")
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_dendrogram(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    method: str = "ward",
    orientation: str = "top",
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """Hierarchical clustering dendrogram."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    apply_standard_theme()
    if columns is None:
        columns = get_numeric_columns(df)
    validate_columns_exist(df, columns)
    X = df[columns].dropna().values
    Z = linkage(X, method=method)
    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(Z, ax=ax, orientation=orientation, leaf_rotation=90, leaf_font_size=7, color_threshold=0.7 * max(Z[:, 2]))
    format_axis_labels(ax, title=title or f"Dendrogram ({method.capitalize()} linkage)")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 10: LARGE DATASET VISUALIZATION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_large_dataset_scatter_aggregation(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2-D hexbin aggregation scatter chart suitable for millions of rows.

    Uses matplotlib hexbin when datashader is unavailable.

    Parameters
    ----------
    x_column : x-axis numeric column
    y_column : y-axis numeric column
    title : chart title
    save_path : optional file path

    Returns
    -------
    matplotlib Figure
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column])
    apply_standard_theme()
    sub = df[[x_column, y_column]].dropna()
    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(sub[x_column], sub[y_column], gridsize=50, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")
    format_axis_labels(ax, xlabel=x_column, ylabel=y_column, title=title or f"Aggregated Scatter – {y_column} vs {x_column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_large_dataset_density_visualization(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """KDE density plot optimised for large datasets using matplotlib."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column])
    apply_standard_theme()
    sub = df[[x_column, y_column]].dropna().sample(min(50_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.kdeplot(data=sub, x=x_column, y=y_column, cmap="Blues", fill=True, thresh=0.05, ax=ax)
    format_axis_labels(ax, xlabel=x_column, ylabel=y_column, title=title or f"Density – {y_column} vs {x_column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_scalable_heatmap(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    bins: int = 50,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """2-D histogram heatmap for large datasets."""
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column])
    apply_standard_theme()
    sub = df[[x_column, y_column]].dropna()
    fig, ax = plt.subplots(figsize=(10, 7))
    h, xedges, yedges, img = ax.hist2d(sub[x_column], sub[y_column], bins=bins, cmap="inferno")
    plt.colorbar(img, ax=ax, label="Count")
    format_axis_labels(ax, xlabel=x_column, ylabel=y_column, title=title or f"Density Heatmap – {y_column} vs {x_column}")
    apply_grid_style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 11: INTERACTIVE VISUALIZATION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_interactive_scatter_plot(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    hover_columns: Optional[list[str]] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """
    Fully interactive Plotly scatter with hover tooltips and zoom.

    Parameters
    ----------
    x_column : x-axis column
    y_column : y-axis column
    color_column : optional colour encoding column
    size_column : optional size encoding column
    hover_columns : columns shown in tooltip
    title : chart title
    save_path : optional HTML export path

    Returns
    -------
    plotly Figure
    """
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    cols = [x_column, y_column] + [c for c in (color_column, size_column) if c]
    validate_columns_exist(df, cols)
    fig = px.scatter(
        df, x=x_column, y=y_column, color=color_column, size=size_column,
        hover_data=hover_columns or df.columns.tolist(),
        title=title or f"Interactive: {y_column} vs {x_column}",
        template="plotly_white", opacity=0.75,
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(title_font_size=16, hovermode="closest")
    _save(fig, save_path)
    # return fig


@tool
def create_interactive_time_series(
    state: Annotated[ AgentState, InjectedState],
    date_column: str,
    value_columns: Union[str, list[str]],
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Plotly time series with range-slider and range-selector buttons."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    if isinstance(value_columns, str):
        value_columns = [value_columns]
    validate_columns_exist(df, [date_column] + value_columns)
    dfc = df.copy()
    dfc[date_column] = pd.to_datetime(dfc[date_column])
    dfc = dfc.sort_values(date_column)
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, col in enumerate(value_columns):
        fig.add_trace(go.Scatter(x=dfc[date_column], y=dfc[col], name=col,
                                  line=dict(color=palette[i % len(palette)], width=1.5), mode="lines"))
    fig.update_layout(
        title=title or "Interactive Time Series",
        template="plotly_white", title_font_size=16, hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ]),
            rangeslider=dict(visible=True), type="date",
        ),
    )
    _save(fig, save_path)
    # return fig


@tool
def create_hover_enabled_bar_chart(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Interactive bar chart with hover tooltips."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column])
    fig = px.bar(
        df, x=x_column, y=y_column, color=color_column,
        hover_data=df.columns.tolist(),
        title=title or f"Bar Chart – {y_column} by {x_column}",
        template="plotly_white", barmode="group",
    )
    fig.update_layout(title_font_size=16)
    _save(fig, save_path)
    # return fig


@tool
def create_zoomable_heatmap(
    state: Annotated[ AgentState, InjectedState],
    x_column: str,
    y_column: str,
    value_column: str,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> go.Figure:
    """Interactive zoomable heatmap via Plotly."""
    save_path = f"visualizations/{title}.html"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    validate_columns_exist(df, [x_column, y_column, value_column])
    pivot = df.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc="mean")
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdBu_r", zmid=0,
    ))
    fig.update_layout(title=title or f"Zoomable Heatmap – {value_column}", template="plotly_white", title_font_size=16)
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 12: DATA QUALITY VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_missing_value_heatmap(
    state: Annotated[ AgentState, InjectedState],
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of missing values across all columns.

    Parameters
    ----------
    title : chart title
    save_path : optional file path

    Returns
    -------
    matplotlib Figure
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    apply_standard_theme()
    missing_matrix = df.isnull().astype(int)
    fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 0.7), min(12, max(5, len(df) * 0.02))))
    sns.heatmap(missing_matrix, cmap="YlOrRd", cbar=True, ax=ax,
                yticklabels=False, xticklabels=True, linewidths=0)
    format_axis_labels(ax, title=title or "Missing Value Heatmap (Yellow = Missing)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_outlier_detection_plot(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    method: str = "zscore",
    threshold: float = 3.0,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Box plots with outlier annotations using Z-score or IQR detection.

    Parameters
    ----------
    columns : numeric columns to analyse (defaults to all)
    method : 'zscore' | 'iqr'
    threshold : Z-score threshold or IQR multiplier
    title : chart title
    save_path : optional file path

    Returns
    -------
    matplotlib Figure
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df= state["clean_df"]
    apply_standard_theme()
    if columns is None:
        columns = get_numeric_columns(df)
    validate_columns_exist(df, columns)
    n = len(columns)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = np.array(axes).flatten()
    for ax, col in zip(axes, columns):
        series = df[col].dropna()
        if method == "zscore":
            z = np.abs(stats.zscore(series))
            mask = z > threshold
        else:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            mask = (series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)
        sns.boxplot(y=series, ax=ax, color="#4C72B0", width=0.4, linewidth=1)
        outliers = series[mask]
        ax.scatter([0] * len(outliers), outliers, color="crimson", zorder=5, s=20, alpha=0.7, label=f"{mask.sum()} outliers")
        ax.legend(fontsize=7)
        format_axis_labels(ax, ylabel=col, title=col)
        apply_grid_style(ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title or f"Outlier Detection ({method.upper()}, threshold={threshold})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


@tool
def create_distribution_comparison_before_after(
    state: Annotated[ AgentState, InjectedState],
    columns: Optional[list[str]] = None,
    title: Optional[str] = None,
    # save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side KDE comparison of distributions before and after cleaning.

    Parameters
    ----------
    columns : numeric columns to compare (defaults to shared numeric cols)
    title : chart title
    save_path : optional file path

    Returns
    -------
    matplotlib Figure
    """
    save_path = f"visualizations/{title}.png"
    remove_first(state["tool_priority_list_3"])
    df_before= state["df"]
    df_after = state["clean_df"]
    apply_standard_theme()
    if columns is None:
        num_before = set(get_numeric_columns(df_before))
        num_after = set(get_numeric_columns(df_after))
        columns = list(num_before & num_after)
    n = len(columns)
    if n == 0:
        raise ValueError("No overlapping numeric columns found.")
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = np.array(axes).flatten()
    for ax, col in zip(axes, columns):
        sns.kdeplot(df_before[col].dropna(), ax=ax, color="#C44E52", label="Before", fill=True, alpha=0.3)
        sns.kdeplot(df_after[col].dropna(), ax=ax, color="#4C72B0", label="After", fill=True, alpha=0.3)
        ax.legend(fontsize=7)
        format_axis_labels(ax, xlabel=col, ylabel="Density", title=col)
        apply_grid_style(ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title or "Distribution: Before vs After Cleaning", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    # return fig


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY  (for AI agent integration)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    # ── Univariate ──────────────────────────────────────────────────────────
    "create_histogram": create_histogram,
    "create_kde_plot": create_kde_plot,
    "create_box_plot": create_box_plot,
    "create_violin_plot": create_violin_plot,
    "create_frequency_bar_chart": create_frequency_bar_chart,
    "create_pie_chart": create_pie_chart,
    # ── Bivariate ───────────────────────────────────────────────────────────
    "create_scatter_plot": create_scatter_plot,
    "create_regression_plot": create_regression_plot,
    "create_grouped_bar_chart": create_grouped_bar_chart,
    "create_box_plot_by_category": create_box_plot_by_category,
    "create_violin_plot_by_category": create_violin_plot_by_category,
    "create_categorical_comparison_chart": create_categorical_comparison_chart,
    # ── Multivariate ────────────────────────────────────────────────────────
    "create_pair_plot": create_pair_plot,
    "create_correlation_heatmap": create_correlation_heatmap,
    "create_bubble_chart": create_bubble_chart,
    "create_grouped_scatter_plot": create_grouped_scatter_plot,
    "create_stacked_bar_chart": create_stacked_bar_chart,
    "create_cluster_heatmap": create_cluster_heatmap,
    "create_parallel_coordinates_plot": create_parallel_coordinates_plot,
    "create_radar_chart": create_radar_chart,
    # ── Time Series ─────────────────────────────────────────────────────────
    "create_time_series_line_chart": create_time_series_line_chart,
    "create_moving_average_chart": create_moving_average_chart,
    "create_seasonal_decomposition_plot": create_seasonal_decomposition_plot,
    "create_time_series_comparison_chart": create_time_series_comparison_chart,
    "create_time_series_area_chart": create_time_series_area_chart,
    # ── Statistical ─────────────────────────────────────────────────────────
    "create_correlation_matrix_heatmap": create_correlation_matrix_heatmap,
    "create_regression_analysis_visualization": create_regression_analysis_visualization,
    "create_residual_plot": create_residual_plot,
    "create_distribution_comparison_chart": create_distribution_comparison_chart,
    "create_statistical_significance_visualization": create_statistical_significance_visualization,
    # ── Geospatial ──────────────────────────────────────────────────────────
    "create_choropleth_map": create_choropleth_map,
    "create_geospatial_scatter_map": create_geospatial_scatter_map,
    "create_location_density_heatmap": create_location_density_heatmap,
    "create_regional_comparison_map": create_regional_comparison_map,
    # ── Network ─────────────────────────────────────────────────────────────
    "create_node_link_graph": create_node_link_graph,
    "create_dependency_graph": create_dependency_graph,
    "create_relationship_network_graph": create_relationship_network_graph,
    # ── Dimensionality Reduction ────────────────────────────────────────────
    "create_pca_visualization": create_pca_visualization,
    "create_tsne_plot": create_tsne_plot,
    "create_umap_plot": create_umap_plot,
    # ── Hierarchical ────────────────────────────────────────────────────────
    "create_treemap": create_treemap,
    "create_sunburst_chart": create_sunburst_chart,
    "create_dendrogram": create_dendrogram,
    # ── Large Dataset ───────────────────────────────────────────────────────
    "create_large_dataset_scatter_aggregation": create_large_dataset_scatter_aggregation,
    "create_large_dataset_density_visualization": create_large_dataset_density_visualization,
    "create_scalable_heatmap": create_scalable_heatmap,
    # ── Interactive ─────────────────────────────────────────────────────────
    "create_interactive_scatter_plot": create_interactive_scatter_plot,
    "create_interactive_time_series": create_interactive_time_series,
    "create_hover_enabled_bar_chart": create_hover_enabled_bar_chart,
    "create_zoomable_heatmap": create_zoomable_heatmap,
    # ── Data Quality ────────────────────────────────────────────────────────
    "create_missing_value_heatmap": create_missing_value_heatmap,
    "create_outlier_detection_plot": create_outlier_detection_plot,
    "create_distribution_comparison_before_after": create_distribution_comparison_before_after

}


# @tool
# def call_tool(tool_name: str, **kwargs) -> Any:
#     """
#     Dispatch a visualization tool call by name.

#     Parameters
#     ----------
#     tool_name : key in TOOL_REGISTRY
#     **kwargs : keyword arguments forwarded to the tool function

#     Returns
#     -------
#     Whatever the individual tool function returns.

#     Raises
#     ------
#     KeyError if *tool_name* is not registered.
#     """
#     if tool_name not in TOOL_REGISTRY:
#         available = ", ".join(sorted(TOOL_REGISTRY))
#         raise KeyError(f"Unknown tool '{tool_name}'. Available tools:\n{available}")
#     return TOOL_REGISTRY[tool_name](**kwargs)