# nodes/LoadDataset.py

from utils.state import AgentState
from utils.dataframe_store import save_df
import pandas as pd
import os


def Load_file(state: AgentState) -> dict:

    file_path = state["file_path"]

    ext = os.path.splitext(file_path)[-1].lower()
    file_type = {
        ".csv":   "csv",
        ".tsv":   "csv",
        ".xlsx":  "excel",
        ".xls":   "excel",
        ".json":  "json",
        ".jsonl": "json",
    }.get(ext, None)

    if file_type == "csv":
        from pathlib import Path
        sep = "\t" if Path(file_path).suffix == ".tsv" else ","
        df = pd.read_csv(file_path, encoding="utf-8", sep=sep)
    elif file_type == "excel":
        df = pd.read_excel(file_path, sheet_name=0)
    elif file_type == "json":
        df = pd.read_json(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(df.shape)

    # Store both raw and clean (initially identical) in the side store
    thread_id = str(file_path)          # unique-enough key per session
    df_key       = f"df_{thread_id}"
    clean_df_key = f"clean_df_{thread_id}"

    save_df(df_key, df)
    save_df(clean_df_key, df.copy())    # clean_df starts as a copy of raw df

    # Return only serializable values — no DataFrames in state
    return {
        "df_key":       df_key,
        "clean_df_key": clean_df_key,
    }