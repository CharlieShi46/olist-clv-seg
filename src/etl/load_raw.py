from __future__ import annotations
import os
import pandas as pd
from src.common.io import read_config, load_csv, save_parquet
from src.common.logging import get_logger

logger = get_logger("etl.load_raw")

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "review_creation_date",
    "review_answer_timestamp",
]

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

def main():
    cfg = read_config()
    paths = cfg["data"]["olist_paths"]

    logger.info("Loading raw csvs...")
    orders = parse_dates(load_csv(paths["orders"]))
    customers = load_csv(paths["customers"])
    items = load_csv(paths["order_items"])
    payments = load_csv(paths["payments"])
    reviews = parse_dates(load_csv(paths["reviews"]))
    products = load_csv(paths["products"])
    cat_map = load_csv(paths["category_map"])
    sellers = load_csv(paths["sellers"])
    geoloc = load_csv(paths["geolocation"])

    out_dir = cfg["data"]["raw_dir"]
    logger.info("Saving standardized parquet to raw_dir...")
    orders.to_parquet(os.path.join(out_dir, "orders.parquet"), index=False)
    customers.to_parquet(os.path.join(out_dir, "customers.parquet"), index=False)
    items.to_parquet(os.path.join(out_dir, "order_items.parquet"), index=False)
    payments.to_parquet(os.path.join(out_dir, "payments.parquet"), index=False)
    reviews.to_parquet(os.path.join(out_dir, "reviews.parquet"), index=False)
    products.to_parquet(os.path.join(out_dir, "products.parquet"), index=False)
    cat_map.to_parquet(os.path.join(out_dir, "category_map.parquet"), index=False)
    sellers.to_parquet(os.path.join(out_dir, "sellers.parquet"), index=False)
    geoloc.to_parquet(os.path.join(out_dir, "geolocation.parquet"), index=False)
    logger.info("Done.")

if __name__ == "__main__":
    print(">>> running load_raw.main() ...")  # 显式提示
    main()
    print(">>> load_raw completed ✅")