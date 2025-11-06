from __future__ import annotations
import os, json
import pandas as pd
import yaml
from dotenv import load_dotenv

def read_config(env: str = "dev") -> dict:
    load_dotenv()
    cfg_path = os.path.join("config", "base.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 可按需合并 dev/prod；当前仅 base 即可
    return cfg

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)