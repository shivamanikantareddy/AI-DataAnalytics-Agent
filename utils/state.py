from typing import TypedDict, Dict, Any, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pathlib import Path
import pandas as pd
import operator


# ── Reducers ────────────────────────────────────────────────────────────────

def merge_dicts_shallow(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level key merge — new values overwrite existing ones at the top level."""
    return {**existing, **new}


def merge_dicts_deep(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive deep merge. Nested dicts are merged rather than overwritten.
    Useful for accumulating analysis_results across multiple nodes.
    """
    merged = dict(existing)
    for key, value in new.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts_deep(merged[key], value)
        else:
            merged[key] = value
    return merged


def append_chat_turns(
    existing: List[Dict[str, str]],
    new: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Appends new chat turns (dicts) to history. Compatible with operator.add."""
    return existing + new


# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    messages:             Annotated[List[BaseMessage], add_messages]
    file_path:            str          # ← str, not Path (Path isn't serializable either)

    # ── DataFrames stored externally, referenced by key ──
    df_key:               str          # key into dataframe_store for raw df
    clean_df_key:         str          # key into dataframe_store for cleaned df

    # ── Cleaning stage ──
    report:               Dict[str, Any]
    cleaning_summary:     Annotated[Dict[str, Any], merge_dicts_shallow]
    tool_priority_list_1: List[Dict[str, Any]]

    # ── EDA stage ──
    eda_result:           Dict[str, Any]
    eda_summary:          Dict[str, Any]
    tool_priority_list_2: List[Dict[str, Any]]

    # ── Analysis stage ──
    analysis_results:     Annotated[Dict[str, Any], merge_dicts_deep]
    tool_priority_list_3: List[Dict[str, Any]]

    # ── Chat stage ──
    chat_history:         Annotated[List[Dict[str, str]], append_chat_turns]
    chat_active:          bool