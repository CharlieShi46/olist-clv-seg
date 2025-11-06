from __future__ import annotations
import pandas as pd, numpy as np, os
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("uplift.eval")

def qini_auc(uplift_scores, treat, y):
    # 排序（从高到低），计算 Qini 曲线与 AUC（简化版）
    order = np.argsort(-uplift_scores)
    y, treat = np.asarray(y)[order], np.asarray(treat)[order]
    cum_t = treat.cumsum()
    cum_c = (~treat.astype(bool)).cumsum()
    # 累积响应(简化：直接用 revenue)
    cum_y_t = (y * treat).cumsum()
    cum_y_c = (y * (~treat.astype(bool))).cumsum()
    # 估计 counterfactual：扩大控制组到同样样本数
    cf = cum_y_c * (cum_t / np.maximum(cum_c,1))
    qini = cum_y_t - cf
    auc = np.trapz(qini, dx=1.0/len(qini))
    return auc, qini

def main():
    cfg = read_config()
    alloc = pd.read_parquet("data/outputs/exp_alloc_demo.parquet")

    # 用 ML 预测的 CLV 作为 uplift proxy（仅用于排序；真实生产应训练 uplift 模型）
    preds = pd.read_parquet(os.path.join(cfg["data"]["features_dir"], "customer_clv_ml.parquet"))
    df = alloc.merge(preds, on="customer_unique_id", how="left").dropna(subset=["clv_ml_pred"])
    df["treat"] = (df["assign"]=="T").astype(int)

    # 假设 outcome（演示用）：未来窗口真实净额（历史替代），或用 clv_ml_pred 加上噪声
    # 如果有真实 outcome，请替换这里：
    outcome = df["clv_ml_pred"] + np.random.default_rng(0).normal(0, df["clv_ml_pred"].std()*0.2, len(df))
    outcome = np.clip(outcome, a_min=0, a_max=None)
    df["revenue"] = outcome

    auc, qini = qini_auc(df["clv_ml_pred"].values, df["treat"].values, df["revenue"].values)
    logger.info(f"Qini AUC (demo): {auc:.2f}")

    # 估算增量利润（粗略）：增量收入 - 成本
    # 按阈值（Top 20%）选择投放
    thr = df["clv_ml_pred"].quantile(0.80)
    cand = df[df["clv_ml_pred"]>=thr]
    inc = cand[cand["treat"]==1]["revenue"].sum() - (cand[cand["treat"]==0]["revenue"].sum() * (cand["treat"].mean()/(1-cand["treat"].mean()+1e-9)))
    cost = cand["expected_cost"].sum()
    profit = inc - cost
    logger.info(f"Incremental Profit (demo): {profit:.2f}")

if __name__ == "__main__":
    main()