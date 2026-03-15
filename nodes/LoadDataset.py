
from utils.state import AgentState
import pandas as pd
import os

    
    
def Load_file( state : AgentState ) -> AgentState:

    file_path = state["file_path"]
    
    df = state["df"]

    ext = os.path.splitext(file_path)[-1].lower()
    file_type = {
        ".csv": "csv",
        ".tsv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".json": "json",
        ".jsonl": "json",
    }.get(ext, None)

    if file_type == "csv":
        sep = "\t" if file_path.suffix == ".tsv" else ","
        df = pd.read_csv(file_path, encoding="utf-8", sep=sep)
    elif file_type == "excel":
        # df:dict[str, DataFrame]
        # sheet_names = pd.ExcelFile(file_path).sheet_names
        # df = pd.read_excel(file_path, sheet_name=None)

        df = pd.read_excel(file_path, sheet_name=0)
    elif file_type == "json":
        df = pd.read_json(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file_type: {ext}")
    
    print(df.shape)

    return {'df': df , 'clean_df': df}
    
