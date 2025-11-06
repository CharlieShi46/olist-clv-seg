from __future__ import annotations
import os, numpy as np, pandas as pd
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("uplift.design")

def main():
    cfg = read_config()
    # 读取客户标签表（上一节生成）
    labeled = pd.read_parquet("data/outputs/CLV_customer_labeled.parquet")

    # 分层：Segment × CLV_tier
    strata = labeled.groupby(["segment_kmeans","CLV_tier"])

    alloc = []
    rng = np.random.default_rng(42)
    for (seg, tier), g in strata:
        n = len(g)
        # 例：按层 50% 进 Treatment（根据预算可调）
        t_mask = rng.random(n) < 0.5
        block = g.copy()
        block["assign"] = np.where(t_mask, "T", "C")
        # 策略示例：根据 tier 给不同 policy_code（真实可接入你的策略映射）
        block["policy_code"] = np.select(
            [
                block["CLV_tier"].eq("Top 5%"),
                block["CLV_tier"].eq("Top 20%"),
            ],
            ["VIP_FREE_SHIP", "STEP_DISCOUNT"],
            default="LIGHT_EDM"
        )
        # 预计成本（测试用；真实可按券额/免邮成本）
        block["expected_cost"] = np.select(
            [
                block["policy_code"].eq("VIP_FREE_SHIP"),
                block["policy_code"].eq("STEP_DISCOUNT")
            ],
            [5.0, 2.0],
            default=0.2
        )
        alloc.append(block[["customer_unique_id","segment_kmeans","CLV_tier","assign","policy_code","expected_cost"]])

    out = pd.concat(alloc, ignore_index=True)
    os.makedirs("data/outputs", exist_ok=True)
    out_path = "data/outputs/exp_alloc_demo.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"✅ Saved allocation: {out_path} | rows={len(out)}")

if __name__ == "__main__":
    main()