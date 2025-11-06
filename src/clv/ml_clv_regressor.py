from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("clv.ml")

DEFAULT_FEATURES = [
    "Recency","Frequency","Monetary",
    "lifetime_days","avg_purchase_interval",
    "avg_review_score","bad_review_rate",
    "avg_installments","credit_share","payment_variety"
]

def train_single_ref(cfg, ref_date: str, features: list[str] = None) -> Dict:
    """
    单一参考日训练一次：用 ref_date 之前的特征做 XGB 回归，标签是 (ref, ref+180] 未来消费额。
    返回 model 与评估指标。
    """
    from src.clv.objectives import make_labels

    feat_path = os.path.join(cfg["data"]["features_dir"], "customer_features.parquet")
    feats = pd.read_parquet(feat_path)

    labels = make_labels(cfg, ref_date, cfg["experiment"]["pred_window_days"])
    df = feats.merge(labels[["customer_unique_id","future_spend"]], on="customer_unique_id", how="left").fillna({"future_spend":0.0})

    # 训练集（只要特征，无需时间过滤，因为特征本身是 ref_date 的静态快照）
    if features is None:
        features = DEFAULT_FEATURES.copy()

    X = df[features].values
    y = df["future_spend"].values

    # 简单基线模型（可后续换 Optuna 调参）
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rho, _ = spearmanr(y, y_pred)
    logger.info(f"[{ref_date}] Train MAE={mae:.2f}, Spearman={rho:.3f}")

    return {
        "model": model,
        "features": features,
        "df_train": df,      # 可用于分析
        "metrics": {"mae": mae, "spearman": float(rho)}
    }

def evaluate_holdout(cfg, model, ref_date: str, features: list[str]) -> Dict:
    """
    （可选）对同一天做自评；如果你用 rolling refs，可以取最近一次作为“近似外推”检验
    这里先返回训练集评估，后面在 train_clv_ml.py 里用多参考日回溯。
    """
    from src.clv.objectives import make_labels
    feats = pd.read_parquet(os.path.join(cfg["data"]["features_dir"], "customer_features.parquet"))
    labels = make_labels(cfg, ref_date, cfg["experiment"]["pred_window_days"])
    df = feats.merge(labels[["customer_unique_id","future_spend"]], on="customer_unique_id", how="left").fillna({"future_spend":0.0})
    X = df[features].values
    y = df["future_spend"].values
    y_pred = model.predict(X)

    from sklearn.metrics import mean_absolute_error
    from scipy.stats import spearmanr
    mae = mean_absolute_error(y, y_pred)
    rho, _ = spearmanr(y, y_pred)

    return {"mae": mae, "spearman": float(rho)}