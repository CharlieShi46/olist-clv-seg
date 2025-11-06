import pandas as pd
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("features.review")

def compute_review_features(cfg):
    reviews = pd.read_parquet(cfg["data"]["raw_dir"] + "/reviews.parquet")
    orders = pd.read_parquet(cfg["data"]["raw_dir"] + "/orders.parquet")
    customers = pd.read_parquet(cfg["data"]["raw_dir"] + "/customers.parquet")

    # 合并得到 customer_unique_id
    cust_map = customers[["customer_id","customer_unique_id"]]
    rev = reviews.merge(orders[["order_id","customer_id"]], on="order_id", how="left")
    rev = rev.merge(cust_map, on="customer_id", how="left")

    agg = rev.groupby("customer_unique_id").agg(
        avg_review_score=("review_score","mean"),
        bad_review_rate=("review_score", lambda s: (s <= 2).mean())
    ).reset_index()

    return agg

def main():
    cfg = read_config()
    out = compute_review_features(cfg)
    out_path = cfg["data"]["features_dir"] + "/review_features.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path} | shape={out.shape}")

if __name__ == "__main__":
    main()