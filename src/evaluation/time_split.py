from __future__ import annotations
import pandas as pd
from typing import List
from src.common.logging import get_logger

logger = get_logger("eval.time_split")

def rolling_ref_dates(end_date: str, n_splits: int = 3, step_months: int = 3) -> list[str]:
    """
    生成一组向过去滚动的参考日（例如 2018-08-31, 2018-05-31, 2018-02-28 ...）
    """
    ref = pd.to_datetime(end_date, utc=True)
    refs = [ref]
    cur = ref
    for _ in range(n_splits - 1):
        cur = cur - pd.DateOffset(months=step_months)
        refs.append(cur)
    refs = [d.strftime("%Y-%m-%d") for d in refs]
    logger.info(f"CV refs: {refs}")
    return refs