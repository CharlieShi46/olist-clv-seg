import pandas as pd
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("features.payment")

def compute_payment_features(cfg):
    payments = pd.read_parquet(cfg["data"]["raw_dir"] + "/payments.parquet")
    orders = pd.read_parquet(cfg["data"]["raw_dir"] + "/orders.parquet")
    customers = pd.read_parquet(cfg["data"]["raw_dir"] + "/customers.parquet")

    cust_map = customers[["customer_id","customer_unique_id"]]
    pay = payments.merge(orders[["order_id","customer_id"]], on="order_id", how="left")
    pay = pay.merge(cust_map, on="customer_id", how="left")

    agg = pay.groupby("customer_unique_id").agg(
        avg_installments=("payment_installments","mean"),
        credit_share=("payment_type", lambda s: (s=="credit_card").mean()),
        payment_variety=("payment_type", lambda s: s.nunique())
    ).reset_index()

    return agg

def main():
    cfg = read_config()
    out = compute_payment_features(cfg)
    out_path = cfg["data"]["features_dir"] + "/payment_features.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path} | shape={out.shape}")

if __name__ == "__main__":
    main()