from __future__ import annotations
import os
import pandas as pd
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("etl.clean_transform")

def load_parquet(path):
    return pd.read_parquet(path)

def merge_line_items(cfg) -> pd.DataFrame:
    raw_dir = cfg["data"]["raw_dir"]
    orders = load_parquet(os.path.join(raw_dir, "orders.parquet"))
    items = load_parquet(os.path.join(raw_dir, "order_items.parquet"))
    payments = load_parquet(os.path.join(raw_dir, "payments.parquet"))

    # 订单-明细
    line = items.merge(orders, on="order_id", how="left", suffixes=("", "_ord"))
    # 订单-支付（聚合到 order 级）
    pay_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(total_payment_value=("payment_value", "sum"),
             payment_types=("payment_type", lambda s: ",".join(sorted(s.astype(str).unique()))),
             installments=("payment_installments", "max"))
    )
    line = line.merge(pay_agg, on="order_id", how="left")

    # 简单异常处理示例：移除缺少客户ID/缺少purchase时间的记录
    line = line.dropna(subset=["customer_id"])
# 统一购买时间列名
    if "order_purchase_timestamp_ord" in line.columns:
        line = line.rename(columns={"order_purchase_timestamp_ord": "purchase_ts"})
    else:
        line = line.rename(columns={"order_purchase_timestamp": "purchase_ts"})

    line["gross_item_value"] = line["price"] + line.get("freight_value", 0) # 毛利口径
    return line

def main():
    cfg = read_config()
    logger.info("Merging to line-items...")
    line = merge_line_items(cfg)
    out_path = os.path.join(cfg["data"]["interim_dir"], "line_items.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    line.to_parquet(out_path, index=False)
    logger.info(f"Saved: {out_path} | shape={line.shape}")

if __name__ == "__main__":
    main()