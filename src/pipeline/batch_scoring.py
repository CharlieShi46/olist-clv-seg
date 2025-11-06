import os, json, glob
from datetime import datetime, timezone
import pandas as pd
from xgboost import XGBRegressor
from src.common.io import read_config
from src.common.logging import get_logger
from src.clv.ml_clv_regressor import DEFAULT_FEATURES

logger = get_logger("pipeline.batch_scoring")

def main():
    cfg = read_config()

    # ✅ 优先读取 meta 文件，确定最佳模型
    meta_files = glob.glob("models/xgb_clv_*_meta.json")
    if not meta_files:
        raise FileNotFoundError("No model metadata found under models/")
    best_meta_file = sorted(meta_files)[-1]
    with open(best_meta_file) as f:
        meta = json.load(f)
    best_ref = meta["ref"]
    model_path = f"models/xgb_clv_{best_ref}.json"
    logger.info(f"✅ Using best-performing model ({best_ref}) from {best_meta_file}")

    # ✅ 加载模型与特征
    features = json.load(open("models/xgb_clv_features.json"))
    model = XGBRegressor()
    model.load_model(model_path)
    logger.info(f"✅ Loaded model: {model_path}")

    # ✅ 读取最新特征表
    feats_path = os.path.join(cfg["data"]["features_dir"], "customer_features.parquet")
    feats = pd.read_parquet(feats_path)
    logger.info(f"Loaded features: {feats_path} | shape={feats.shape}")

    # ✅ 预测与毛利修正
    preds = model.predict(feats[features].values)
    feats["clv_ml_pred"] = preds * cfg["model"]["margin_rate"]

    # ✅ 保存输出文件
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = f"data/outputs/CLV_ML_{ts}.parquet"
    os.makedirs("data/outputs", exist_ok=True)
    feats[["customer_unique_id", "clv_ml_pred"]].to_parquet(out_path, index=False)
    logger.info(f"✅ Batch scoring done: {out_path} | rows={len(feats)}")

if __name__ == "__main__":
    main()