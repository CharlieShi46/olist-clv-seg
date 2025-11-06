import pandas as pd
import hdbscan
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.common.io import read_config
from src.common.logging import get_logger
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

logger = get_logger("segmentation.hdbscan")

def train_hdbscan(df, feature_cols):
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=10)
    labels = clusterer.fit_predict(X_scaled)
    n_clusters = len(np.unique(labels[labels>=0]))
    logger.info(f"✅ HDBSCAN clusters={n_clusters}, noise={(labels==-1).mean():.2%}")
    return labels, clusterer

def visualize_clusters(df, feature_cols, labels, out_path="data/features/hdbscan_clusters.png"):
    X = StandardScaler().fit_transform(df[feature_cols])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    df["x"], df["y"] = coords[:,0], coords[:,1]
    df["label"] = labels
    plt.figure(figsize=(6,5))
    plt.scatter(df["x"], df["y"], c=df["label"], cmap="tab10", s=10)
    plt.title("HDBSCAN Clusters (PCA 2D)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cluster plot: {out_path}")

def main():
    cfg = read_config()
    df = pd.read_parquet(cfg["data"]["features_dir"] + "/customer_features.parquet")
    feature_cols = ["Recency","Frequency","Monetary","avg_review_score","credit_share","bad_review_rate"]

    labels, clusterer = train_hdbscan(df, feature_cols)
    df["segment_hdbscan"] = labels

    out_path = cfg["data"]["features_dir"] + "/customer_segments_hdbscan.parquet"
    df.to_parquet(out_path, index=False)
    visualize_clusters(df, feature_cols, labels)
    logger.info(f"Saved {out_path}")

if __name__ == "__main__":
    main()