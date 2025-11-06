import pandas as pd
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("features.time")

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """客户生命周期、平均间隔天数"""
    df = df.copy()
    df["lifetime_days"] = (df["last_purchase"] - df["first_purchase"]).dt.days
    df["avg_purchase_interval"] = df["lifetime_days"] / df["orders_count"].clip(lower=1)
    df["avg_purchase_interval"] = df["avg_purchase_interval"].fillna(df["lifetime_days"])
    return df[["customer_unique_id", "lifetime_days", "avg_purchase_interval"]]

def main():
    cfg = read_config()
    df = pd.read_parquet(cfg["data"]["interim_dir"] + "/customer_wide.parquet")
    out = compute_time_features(df)
    out_path = cfg["data"]["features_dir"] + "/time_features.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path} | shape={out.shape}")

if __name__ == "__main__":
    main()