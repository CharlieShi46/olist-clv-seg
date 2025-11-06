from __future__ import annotations
import os, joblib, json, glob
import numpy as np
from xgboost import Booster
import pandas as pd
from typing import List, Dict
from src.common.io import read_config
from src.common.logging import get_logger
from src.evaluation.time_split import rolling_ref_dates
from src.clv.ml_clv_regressor import train_single_ref, DEFAULT_FEATURES

logger = get_logger("pipeline.train_clv_ml")

def main():
    cfg = read_config()
    refs = rolling_ref_dates(cfg["experiment"]["reference_date"], n_splits=3, step_months=3)

    cv_metrics = []
    models = []

    for ref in refs[::-1]:
        out = train_single_ref(cfg, ref, DEFAULT_FEATURES)
        cv_metrics.append({"ref": ref, **out["metrics"]})
        models.append((ref, out["model"]))

    logger.info("CV results:")
    for m in cv_metrics:
        logger.info(m)

    # =====================================
    # ✅ 选择 Spearman 最优模型
    # =====================================
    best_model_info = max(cv_metrics, key=lambda m: m["spearman"])
    best_ref = best_model_info["ref"]
    prod_model = next(model for ref, model in models if ref == best_ref)
    logger.info(f"✅ Using best-performing model ({best_ref}) for production scoring.")
    logger.info(f"MAE={best_model_info['mae']:.4f}, Spearman={best_model_info['spearman']:.4f}")

    os.makedirs("models", exist_ok=True)
    model_json = f"models/xgb_clv_{best_ref}.json"
    prod_model.save_model(model_json)
    logger.info(f"✅ Saved model (xgboost json): {model_json}")

    with open("models/xgb_clv_features.json", "w") as f:
        json.dump(DEFAULT_FEATURES, f, indent=2)

    meta = {"ref": best_ref, "metrics": best_model_info, "features": DEFAULT_FEATURES}
    with open(f"models/xgb_clv_{best_ref}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved model metadata: models/xgb_clv_{best_ref}_meta.json")

    # =====================================
    # ✅ 用最优模型在最新客户特征上打分
    # =====================================
    feat_path = os.path.join(cfg["data"]["features_dir"], "customer_features.parquet")
    logger.info(f"Scoring latest feature set from {feat_path}")
    feats = pd.read_parquet(feat_path)
    X = feats[DEFAULT_FEATURES].values
    y_pred = prod_model.predict(X)

    margin = cfg["model"]["margin_rate"]
    feats["clv_ml_pred"] = y_pred * margin

    out_path = os.path.join(cfg["data"]["features_dir"], "customer_clv_ml.parquet")
    feats[["customer_unique_id","clv_ml_pred"]].to_parquet(out_path, index=False)
    logger.info(f"✅ Saved predictions: {out_path} | shape={feats.shape}")

    with open("models/xgb_clv_cv_metrics.json","w") as f:
        json.dump(cv_metrics, f, indent=2)
    logger.info("Saved CV metrics json.")

if __name__ == "__main__":
    main()