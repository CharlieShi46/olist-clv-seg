from __future__ import annotations
import os
import pandas as pd
from typing import Tuple
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("clv.objectives")

def make_labels(cfg, ref_date: str, horizon_days: int = 180) -> pd.DataFrame:
    """
    为每个 customer_unique_id 构造 [ref_date, ref_date + horizon] 的目标 y（未来消费额）。
    使用 data/interim/line_items.parquet 中的 purchase_ts & gross_item_value。
    """
    ref = pd.to_datetime(ref_date, utc=True)
    horizon_end = ref + pd.Timedelta(days=horizon_days)

    line = pd.read_parquet(os.path.join(cfg["data"]["interim_dir"], "line_items.parquet"))
    customers = pd.read_parquet(os.path.join(cfg["data"]["raw_dir"], "customers.parquet"))

    # map 到 unique id
    cust_map = customers[["customer_id","customer_unique_id"]].drop_duplicates()
    li = line.merge(cust_map, on="customer_id", how="left")

    # 目标窗 (ref, ref+horizon]
    mask = (li["purchase_ts"] > ref) & (li["purchase_ts"] <= horizon_end)
    future = (
        li.loc[mask]
        .groupby("customer_unique_id", as_index=False)
        .agg(future_spend=("gross_item_value", "sum"),
             future_orders=("order_id","nunique"))
    )

    # 没消费的客户也要保留，设为 0
    all_ids = cust_map["customer_unique_id"].drop_duplicates()
    labels = pd.DataFrame({"customer_unique_id": all_ids})
    labels = labels.merge(future, on="customer_unique_id", how="left").fillna({"future_spend":0.0, "future_orders":0})

    labels["ref_date"] = ref
    labels["horizon_days"] = horizon_days
    logger.info(f"Labels built for ref={ref_date} | shape={labels.shape} | spend>0 share={(labels['future_spend']>0).mean():.2%}")
    return labels