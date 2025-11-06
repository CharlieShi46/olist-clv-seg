import pandas as pd
import os
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("features.store")

def main():
    cfg = read_config()
    feat_dir = cfg["data"]["features_dir"]

    rfm = pd.read_parquet(os.path.join(feat_dir, "rfm_features.parquet"))
    timef = pd.read_parquet(os.path.join(feat_dir, "time_features.parquet"))
    review = pd.read_parquet(os.path.join(feat_dir, "review_features.parquet"))
    pay = pd.read_parquet(os.path.join(feat_dir, "payment_features.parquet"))

    df = rfm.merge(timef, on="customer_unique_id", how="left")
    df = df.merge(review, on="customer_unique_id", how="left")
    df = df.merge(pay, on="customer_unique_id", how="left")

    # 简单缺失处理
    df = df.fillna({
        "avg_review_score": df["avg_review_score"].mean(),
        "bad_review_rate": 0,
        "avg_installments": 0,
        "credit_share": 0,
        "payment_variety": 1
    })

    out_path = os.path.join(feat_dir, "customer_features.parquet")
    df.to_parquet(out_path, index=False)
    logger.info(f"✅ Saved final feature table: {out_path} | shape={df.shape}")

if __name__ == "__main__":
    main()