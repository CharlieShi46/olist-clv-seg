import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.common.logging import get_logger
from src.common.io import read_config
import matplotlib.pyplot as plt
import os

logger = get_logger("segmentation.kmeans")

def train_kmeans(df, feature_cols, k_range=(3,10)):
    """自动寻找最佳K并训练KMeans模型"""
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    best_k, best_score = None, -1
    scores = {}

    for k in range(k_range[0], k_range[1]+1):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        logger.info(f"K={k}, silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_model, best_labels = k, score, model, labels

    logger.info(f"✅ Best K={best_k}, silhouette={best_score:.4f}")
    return best_model, best_labels, scores

def plot_silhouette(scores, out_path="data/features/silhouette.png"):
    plt.figure(figsize=(6,4))
    ks, vals = zip(*scores.items())
    plt.plot(ks, vals, marker="o")
    plt.title("Silhouette Score by K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved silhouette plot: {out_path}")

def main():
    cfg = read_config()
    df = pd.read_parquet(cfg["data"]["features_dir"] + "/customer_features.parquet")

    # 选择聚类特征
    feature_cols = ["Recency","Frequency","Monetary","avg_review_score","credit_share","bad_review_rate"]
    logger.info(f"Feature columns: {feature_cols}")

    model, labels, scores = train_kmeans(df, feature_cols, k_range=(3,10))
    plot_silhouette(scores)

    df["segment_kmeans"] = labels
    out_path = cfg["data"]["features_dir"] + "/customer_segments_kmeans.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path} | shape={df.shape}")

if __name__ == "__main__":
    main()