from typing import TypedDict, Dict, Any, Annotated
from pathlib import Path
import pandas as pd


def merge_dicts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    return {**existing, **new}

class AgentState(TypedDict):
    file_path : Path
    df : pd.DataFrame
    report: Dict[str, Any]
    clean_df : pd.DataFrame
    eda_result: Dict[str, Any]
    analysis_results : Annotated[Dict[str, Any], merge_dicts]
