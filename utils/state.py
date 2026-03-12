from typing import TypedDict, Dict, Any
from pathlib import Path
import pandas as pd


class AgentState(TypedDict):
    file_path : Path
    df : pd.DataFrame
    report: Dict[str, Any]
    cleaned_df : pd.DataFrame
    eda_result: Dict[str, Any]
    
