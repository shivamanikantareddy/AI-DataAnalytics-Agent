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

class AgentState(TypedDict):
    # ── LLM message bus (tool calls, AI responses, etc.) ──
    messages: Annotated[List[BaseMessage], add_messages]

    # ── Input data ──
    # NOTE: Path and DataFrame are not serializable by LangGraph checkpointers.
    # Avoid using memory/sqlite checkpointers if these fields are populated.
    file_path: Path
    df: pd.DataFrame

    # ── Cleaning stage ──
    report: Dict[str, Any]                                              # raw profiling report
    cleaning_summary: Annotated[Dict[str, Any], merge_dicts_shallow]   # accumulated cleaning steps
    tool_priority_list_1: List[Dict[str, Any]]                         # cleaning tool queue
    clean_df: pd.DataFrame                                              # cleaned dataframe

    # ── EDA stage ──
    tool_priority_list_2: List[Dict[str, Any]]                         # EDA tool queue
    eda_result: Dict[str, Any]                                          # raw EDA outputs
    eda_summary: Dict[str, Any]                                         # summarized EDA findings

    # ── Analysis stage ──
    tool_priority_list_3: List[Dict[str, Any]]                         # analysis tool queue
    analysis_results: Annotated[Dict[str, Any], merge_dicts_deep]      # deep-merged across nodes

    # ── Chat stage ──
    chat_history: Annotated[List[Dict[str, str]], append_chat_turns]   # [{"user": ..., "assistant": ...}]
    chat_active: bool                                                   # controls chat loop routing