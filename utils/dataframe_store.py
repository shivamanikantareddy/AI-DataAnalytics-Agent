
import pandas as pd
from typing import Dict, Optional

_store: Dict[str, pd.DataFrame] = {}

def save_df(key: str, df: pd.DataFrame) -> None:
    _store[key] = df

def load_df(key: str) -> Optional[pd.DataFrame]:
    return _store.get(key)

def delete_df(key: str) -> None:
    _store.pop(key, None)