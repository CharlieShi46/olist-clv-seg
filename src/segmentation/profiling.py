import pandas as pd
from src.common.io import read_config
from src.common.logging import get_logger

logger = get_logger("segmentation.profiling")

def profile_segments(df, label_col):
    features = ["Recency","Frequency","Monetary","avg_review_score","bad_review_rate","credit_share"]
    profile = (
        df.groupby(label_col)[features]
        .mean()
        .round(2)
        .sort_values("Monetary", ascending=False)
    )
    profile["count"] = df.groupby(label_col).size()
    profile["share"] = profile["count"] / len(df)
    return profile

def main():
    cfg = read_config()
    df = pd.read_parquet(cfg["data"]["features_dir"] + "/customer_segments_kmeans.parquet")
    profile = profile_segments(df, "segment_kmeans")
    print("\n=== Segment Profiles ===\n", profile.head(10))
    out_path = cfg["data"]["features_dir"] + "/segment_profiles.csv"
    profile.to_csv(out_path)
    logger.info(f"Saved {out_path}")

if __name__ == "__main__":
    main()