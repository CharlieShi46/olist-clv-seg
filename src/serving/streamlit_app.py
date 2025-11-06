import os, json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from xgboost import XGBRegressor

FEATURES_DIR = "data/features"
OUTPUTS_DIR = "data/outputs"
MODELS_DIR = "models"

@st.cache_data
def load_data():
    """加载特征 + 分群 + CLV 预测并去重"""
    features_dir = "data/features"
    outputs_dir = "data/outputs"

    # 读取基础数据
    seg = pd.read_parquet(os.path.join(features_dir, "customer_segments_kmeans.parquet"))
    feats = pd.read_parquet(os.path.join(features_dir, "customer_features.parquet"))
    latest = sorted([f for f in os.listdir(outputs_dir) if f.startswith("CLV_ML_")])[-1]
    clv = pd.read_parquet(os.path.join(outputs_dir, latest))

    # 🔧 保留 seg 里只有 customer_id + segment
    seg = seg[["customer_unique_id", "segment_kmeans"]]

    # 🔧 合并，并确保列名唯一
    df = seg.merge(feats, on="customer_unique_id", how="left")
    df = df.merge(clv, on="customer_unique_id", how="left")

    # ---- 去重 & 统一列名 ----
# 先统一后缀
    df.columns = df.columns.str.replace("_x", "", regex=False).str.replace("_y", "", regex=False)

    # 如果仍有重复（例如原表本身重复），保留第一次出现
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        print(f"⚠️  Found duplicate columns after merge: {dup_cols}")
        df = df.loc[:, ~df.columns.duplicated()]

    print("✅ Data loaded | shape=", df.shape, "| unique cols:", df.columns.nunique())
    return df

@st.cache_resource
def load_model():
    # 加载最新的 xgboost json 模型
    model_jsons = [f for f in os.listdir(MODELS_DIR) if f.startswith("xgb_clv_") and f.endswith(".json") and "metrics" not in f and "features" not in f]
    model_jsons.sort()
    model_path = os.path.join(MODELS_DIR, model_jsons[-1])
    features = json.load(open(os.path.join(MODELS_DIR, "xgb_clv_features.json")))
    model = XGBRegressor()
    model.load_model(model_path)
    return model, features

def main():
    st.set_page_config(page_title="Olist CLV & Segmentation", layout="wide")
    st.title("Olist — Customer Segmentation & CLV Dashboard")

    df = load_data()
    model, feature_cols = load_model()

    # Sidebar 过滤
    seg_ids = sorted(df["segment_kmeans"].dropna().unique())
    seg_sel = st.sidebar.multiselect("Segments", seg_ids, default=seg_ids)
    clv_min, clv_max = float(df["clv_ml_pred"].min()), float(df["clv_ml_pred"].max())
    clv_range = st.sidebar.slider("CLV range", min_value=clv_min, max_value=clv_max, value=(clv_min, clv_max))

    view = df[df["segment_kmeans"].isin(seg_sel)]
    view = view[(view["clv_ml_pred"]>=clv_range[0]) & (view["clv_ml_pred"]<=clv_range[1])]

    # 概览 KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(view):,}")
    c2.metric("Avg CLV", f"{view['clv_ml_pred'].mean():.2f}")
    c3.metric("Top 10% share", f"{view.sort_values('clv_ml_pred', ascending=False)['clv_ml_pred'].head(max(1,int(0.1*len(view)))).sum()/view['clv_ml_pred'].sum():.0%}" if len(view)>0 else "n/a")
    c4.metric("Segments Shown", f"{len(seg_sel)}")

    # 散点：Frequency vs Monetary（颜色 = segment）
    # 防止负值影响散点大小
    view["clv_ml_pred_plot"] = view["clv_ml_pred"].clip(lower=0)

    fig = px.scatter(
        view.sample(min(len(view), 5000), random_state=42),
        x="Frequency", y="Monetary",
        color="segment_kmeans",
        size="clv_ml_pred_plot",
        hover_data=["customer_unique_id"]
    )
    fig.update_layout(height=420, title="Frequency vs Monetary (sampled)")
    st.plotly_chart(fig, use_container_width=True)

    # 直方图：CLV 分布
    fig2 = px.histogram(view, x="clv_ml_pred", nbins=50, title="CLV Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    # 客户查询
    st.subheader("Customer Lookup")
    cid = st.text_input("customer_unique_id")
    if cid:
        row = df[df["customer_unique_id"]==cid]
        if row.empty:
            st.warning("Customer not found.")
        else:
            st.write(row[["customer_unique_id","segment_kmeans","clv_ml_pred","Recency","Frequency","Monetary","avg_review_score","bad_review_rate","credit_share"]].head(1))
            # 在线重算（演示，一般按批得分即可）
            X = row[feature_cols].values
            pred = float(model.predict(X)[0])
            st.info(f"On-the-fly CLV (raw model): {pred:.2f}")

    st.caption("Data sources: Olist public dataset | This dashboard is a demo for segmentation & CLV ops.")

if __name__ == "__main__":
    main()