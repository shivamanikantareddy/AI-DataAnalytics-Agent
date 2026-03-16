import numpy as np
import pandas as pd
from typing import Any


def to_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas scalars and containers to
    pure Python types that msgpack can serialize.
    """
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]

    # numpy scalars
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas scalars
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None     # NaN / Inf are not valid JSON/msgpack either

    return obj