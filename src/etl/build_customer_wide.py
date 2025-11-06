from __future__ import annotations
import os
import pandas as pd
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("etl.customer_wide")

def build_customer_wide(cfg) -> pd.DataFrame:
    line = pd.read_parquet(os.path.join(cfg["data"]["interim_dir"], "line_items.parquet"))
    # 连接客户表得到唯一客户ID
    customers = pd.read_parquet(os.path.join(cfg["data"]["raw_dir"], "customers.parquet"))
    orders = pd.read_parquet(os.path.join(cfg["data"]["raw_dir"], "orders.parquet"))

    # orders: customer_id ↔ customers: (customer_id, customer_unique_id)
    cust_map = customers[["customer_id", "customer_unique_id"]].drop_duplicates()
    order_basic = orders[["order_id", "customer_id", "order_purchase_timestamp"]].rename(
    columns={"order_purchase_timestamp": "purchase_ts"}
)

    base = line.merge(order_basic, on=["order_id","customer_id"], how="left")
    base = base.merge(cust_map, on="customer_id", how="left")

    # Ensure a unified purchase timestamp column exists after merges
    candidate_ts_cols = [
        "purchase_ts",
        "purchase_ts_x",
        "purchase_ts_y",
        "order_purchase_timestamp",
        "order_purchase_timestamp_ord",
    ]
    available_ts_cols = [c for c in candidate_ts_cols if c in base.columns]
    if not available_ts_cols:
        raise KeyError("No purchase timestamp column found after merges.")
    unified_ts = None
    for c in available_ts_cols:
        unified_ts = base[c] if unified_ts is None else unified_ts.fillna(base[c])
    base["purchase_ts"] = unified_ts
    # Drop duplicate timestamp variants to avoid confusion
    cols_to_drop = [c for c in ["purchase_ts_x","purchase_ts_y","order_purchase_timestamp","order_purchase_timestamp_ord"] if c in base.columns]
    if cols_to_drop:
        base = base.drop(columns=cols_to_drop)

    # 以 customer_unique_id 聚合客户级别指标（最小集合）
    agg = base.groupby("customer_unique_id").agg(
        first_purchase=("purchase_ts","min"),
        last_purchase=("purchase_ts","max"),
        orders_count=("order_id","nunique"),
        total_gross=("gross_item_value","sum"),
        avg_item_price=("price","mean")
    ).reset_index()

    return agg

def main():
    cfg = read_config()
    df = build_customer_wide(cfg)
    out = os.path.join(cfg["data"]["interim_dir"], "customer_wide.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"Saved: {out} | shape={df.shape}")

if __name__ == "__main__":
    main()