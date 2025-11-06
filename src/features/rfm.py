from __future__ import annotations
import pandas as pd
from datetime import datetime
from src.common.logging import get_logger
from src.common.io import read_config

logger = get_logger("features.rfm")

def compute_rfm(customer_df: pd.DataFrame, ref_date: str) -> pd.DataFrame:
    """计算 Recency / Frequency / Monetary 三要素"""
    ref_date = pd.to_datetime(ref_date, utc=True)

    df = customer_df.copy()
    df["Recency"] = (ref_date - df["last_purchase"]).dt.days
    df["Frequency"] = df["orders_count"]
    df["Monetary"] = df["total_gross"]

    rfm = df[["customer_unique_id", "Recency", "Frequency", "Monetary"]]
    logger.info(f"RFM shape={rfm.shape}")
    return rfm

def main():
    cfg = read_config()
    ref_date = cfg["experiment"]["reference_date"]
    cust = pd.read_parquet(cfg["data"]["interim_dir"] + "/customer_wide.parquet")
    rfm = compute_rfm(cust, ref_date)
    out_path = cfg["data"]["features_dir"] + "/rfm_features.parquet"
    rfm.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path}")

if __name__ == "__main__":
    main()