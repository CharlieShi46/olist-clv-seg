![Python](https://img.shields.io/badge/Python-3.11-blue)
![Model](https://img.shields.io/badge/Model-XGBoost-success)
![License](https://img.shields.io/badge/License-MIT-green)

# 🛍️ Olist Customer Segmentation & CLV Prediction

> End-to-end customer segmentation and lifetime value (CLV) prediction pipeline built on the **Brazilian E-Commerce Public Dataset by Olist**, integrating machine learning, probabilistic modeling, and marketing uplift simulation.

---

## 📘 Project Overview

This project builds a **full-stack data science pipeline** for customer value prediction and segmentation in an e-commerce context.  
Using real Olist transaction data (100K+ orders from 2016–2018), it predicts each customer's **future 180-day CLV**, builds interpretable customer segments, and designs **ROI-driven marketing strategies** based on uplift modeling.

---

## 🧩 Key Features

- **🔄 ETL & Feature Engineering**  
  Consolidated 9 raw Olist tables into 95K customer-level records.  
  Engineered behavioral, RFM, payment, and review-based features.

- **🤖 ML-based CLV Model**  
  Trained an `XGBoostRegressor` to predict future 180-day gross profit,  
  validated through rolling time-based splits (Spearman = **0.89**, MAE = **3.04**).

- **📈 Probabilistic Baseline (BG/NBD + Gamma-Gamma)**  
  Built interpretable benchmark models for audit & financial alignment.

- **🧠 Customer Segmentation**  
  Unsupervised learning via `KMeans` and `HDBSCAN`,  
  yielding actionable customer cohorts by value and frequency.

- **📊 Uplift Simulation & Experimentation**  
  Designed treatment-control allocation by CLV tier,  
  quantified incremental ROI uplift (~+15–20%).

- **📑 Automated Reporting**  
  Generates PowerPoint reports (`python-pptx`) with CLV distribution,  
  segmentation heatmaps, uplift allocation charts, and business insights.

---

## 🧱 Project Architecture

olist-clv-seg/
│
├── src/
│   ├── etl/                     # Data cleaning and merging (9 Olist tables)
│   ├── features/                # Feature engineering (RFM, reviews, payments)
│   ├── pipeline/                # Model training, scoring, segmentation
│   ├── reports/                 # Auto PPT generation and visualization
│   └── common/                  # Logging, config, and utility functions
│
├── config/                      # YAML config files for data paths & params
├── data/                        # (ignored) raw, interim, features, outputs
├── models/                      # Saved XGBoost & BG/NBD models
├── reports/                     # Auto-generated PowerPoint deck
├── requirements.txt
└── README.md

## ⚙️ How to Run
---
```bash
# 1️⃣ Create environment
conda create -n olist-clv python=3.11
conda activate olist-clv
pip install -r requirements.txt

# 2️⃣ Generate features
python -m src.etl.build_customer_wide
python -m src.features.build_customer_features

# 3️⃣ Train ML-based CLV model
python -m src.pipeline.train_clv_ml

# 4️⃣ Batch scoring & segmentation
python -m src.pipeline.batch_scoring
python -m src.pipeline.clv_segmentation_merge

# 5️⃣ Generate automated PowerPoint report
python -m src.reports.generate_ppt


🎯 Business Impact
	•	Identified top 20% customers contributing ~80% of total predicted CLV
	•	Enabled marketing budget reallocation to focus on high-value segments
	•	Simulated uplift experiment showed +15–20% incremental ROI
	•	Established reproducible pipeline for weekly scoring and model retraining

⸻

🧠 Tools & Libraries

Python · Pandas · Scikit-learn · XGBoost · Lifetimes · HDBSCAN · SHAP · Matplotlib · python-pptx · Prefect

⸻

📄 Description (for GitHub short tagline)

Machine learning–driven customer lifetime value prediction and segmentation pipeline for e-commerce marketing optimization.

⸻

👤 Author

Charlie Shi
Data Science & Business Analytics
GitHub: charlieshi46￼

⸻

✅ Next Steps
	1.	Add a project banner or architecture diagram (reports/figures/diagram.png)
	2.	Connect the repo to a requirements.txt badge / CI pipeline
	3.	Optionally host a Streamlit dashboard for interactive demo