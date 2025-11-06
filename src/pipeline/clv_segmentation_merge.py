from __future__ import annotations
import pandas as pd, numpy as np, os
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("pipeline.clv_merge")

def assign_tier(x, q95, q80, q50):
    if x >= q95:
        return "Top 5%"
    elif x >= q80:
        return "Top 20%"
    elif x >= q50:
        return "Mid 50%"
    else:
        return "Bottom 50%"

def main():
    cfg = read_config()

    # === 1. 读取数据 ===
    clv = pd.read_parquet("data/outputs/CLV_ML_20251105.parquet")
    seg = pd.read_parquet(os.path.join(cfg["data"]["features_dir"], "customer_segments_kmeans.parquet"))

    df = seg.merge(clv, on="customer_unique_id", how="left")

    # === 2. 计算分位阈值 ===
    q95, q80, q50 = np.percentile(df["clv_ml_pred"], [95, 80, 50])
    logger.info(f"CLV cutoffs -> 95th={q95:.2f}, 80th={q80:.2f}, 50th={q50:.2f}")

    df["CLV_tier"] = df["clv_ml_pred"].apply(assign_tier, args=(q95, q80, q50))

    # === 3. 统计分层规模 ===
    tier_summary = (
        df.groupby("CLV_tier")["customer_unique_id"]
        .count()
        .div(len(df))
        .rename("share")
        .reset_index()
    )
    logger.info(f"\nTier share:\n{tier_summary}")

    # === 4. 合并 Segment → 策略映射 ===
    mapping = {
        0: "普通一次性客户 → 再购激励",
        1: "不满意客户 → 售后修复/召回",
        2: "高价值忠诚客户 → 会员优先/新品预推",
    }
    df["segment_strategy"] = df["segment_kmeans"].map(mapping)

    # === 5. 汇总策略矩阵 ===
    summary = (
        df.groupby(["segment_kmeans","CLV_tier"])
          .agg(avg_clv=("clv_ml_pred","mean"),
               customers=("customer_unique_id","count"))
          .reset_index()
    )
    out_csv = "data/outputs/CLV_segment_matrix.csv"
    summary.to_csv(out_csv, index=False)
    logger.info(f"✅ Saved matrix to {out_csv} | shape={summary.shape}")

    out_parquet = "data/outputs/CLV_customer_labeled.parquet"
    df.to_parquet(out_parquet, index=False)
    logger.info(f"✅ Saved full customer table to {out_parquet} | rows={len(df)}")

if __name__ == "__main__":
    main()