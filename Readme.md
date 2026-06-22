# ✈ Flight Delay Intelligence System
### Predictive Analytics Platform for Airline Operations

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square&logo=streamlit)
![Dataset](https://img.shields.io/badge/Dataset-BTS%20On--Time%20Performance-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

> A production-grade flight delay intelligence platform built on 30 million domestic US flights. Predicts delay probability, benchmarks airline performance, identifies airport hotspots, and breaks down delay causes — all in an interactive Streamlit dashboard.

---

## 🔗 Live Demo

**[➜ Open Dashboard](#)** ← *(link after deployment)*

---

## 📌 Project Overview

This is not a simple "train a model and print accuracy" project.

The **Flight Delay Intelligence System** is designed to resemble a real analytics product used by airline operations teams. It answers questions like:

- Will this flight be delayed?
- Which airlines are most reliable?
- Which airports are delay hotspots?
- What actually causes delays — weather, carrier, or air traffic?
- Which routes carry the highest delay risk?

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | [BTS Reporting Carrier On-Time Performance](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018) |
| Raw dataset | 30,144,615 flights · 2014–2018 |
| Working sample | 3,000,000 flights (600K stratified per year) |
| Final processed | 2,944,078 flights after cleaning |
| Airlines | 20 carriers |
| Airports | 367 US airports |
| Features | 8 pre-departure features |
| Target | `ARR_DEL15` — arrival delayed 15+ minutes (BTS standard) |

**Why stratified sampling?** Equal representation across all 5 years prevents any single year from dominating model training. The full 30M row dataset is preserved as source of truth.

---

## 🏗 Project Structure

```
flight-delay-intelligence-system/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Dataset inspection & column analysis
│   ├── 02_preprocessing.ipynb        # Cleaning, feature extraction, target creation
│   ├── 03_eda.ipynb                  # Exploratory analysis & delay pattern discovery
│   ├── 04_feature_engineering.ipynb  # Leakage removal, target encoding
│   └── 05_modeling.ipynb             # Model training & evaluation
│
├──  app.py                           # Streamlit multi-page dashboard
│
├── models/
│   └── xgb_model.pkl                 # Trained XGBoost classifier
│
├── data/
│   ├── raw/                          # Original BTS CSV files (not tracked)
│   └── processed/
│       └── flights_processed.parquet # Cleaned dataset for dashboard
│
├── requirements.txt
└── README.md
```

---

## ⚙ ML Workflow

Followed a strict, production-style ML pipeline. No shortcuts.

```
Raw Data (30M rows)
       ↓
Data Understanding    — column types, nulls, date range, unique values
       ↓
Preprocessing         — leakage removal, datetime parsing, cancelled flight separation
       ↓
EDA                   — delay patterns by airline, airport, hour, month, cause
       ↓
Feature Engineering   — target encoding for categoricals, leakage audit
       ↓
Modeling              — baseline → Random Forest → XGBoost
       ↓
Dashboard             — Streamlit analytics platform
```

---

## 🔍 Feature Engineering

**Leakage discipline was strictly enforced.** Any column not available at prediction time was dropped:

| Dropped (Leakage) | Reason |
|---|---|
| `DEP_DELAY`, `ARR_DELAY` | Only known after flight departs |
| `DEP_TIME`, `ARR_TIME` | Actual times — post-event |
| `WHEELS_OFF/ON`, `TAXI_OUT/IN` | Operational data — post-departure |
| `CARRIER/WEATHER/NAS/SECURITY/LATE_AIRCRAFT_DELAY` | Only populated after delay occurs |
| `CRS_ELAPSED_TIME` | 0.98 correlation with DISTANCE — redundant |

**Final features used for prediction:**

| Feature | Type | Description |
|---|---|---|
| `CRS_DEP_TIME` | Numerical | Scheduled departure hour |
| `CRS_ARR_TIME` | Numerical | Scheduled arrival hour |
| `DISTANCE` | Numerical | Route distance in miles |
| `MONTH` | Numerical | Month of flight |
| `DAY_OF_WEEK` | Numerical | Day of week |
| `CARRIER_ENCODED` | Numerical | Airline target-encoded delay rate |
| `ORIGIN_ENCODED` | Numerical | Origin airport target-encoded delay rate |
| `DEST_ENCODED` | Numerical | Destination target-encoded delay rate |

**Target encoding** was chosen over one-hot encoding for `CARRIER`, `ORIGIN`, and `DEST` — one-hot would create 750+ sparse columns and destroy SHAP interpretability.

---

## 🤖 Model Performance

| Model | F1 (Delayed) | Recall (Delayed) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.00 | 0.00 | 0.633 |
| Random Forest | 0.38 | 0.63 | 0.668 |
| **XGBoost (final)** | **0.39** | **0.60** | **0.671** |

**Why not accuracy?** Class distribution is 82% on-time / 18% delayed. A model predicting "never delayed" achieves 82% accuracy — completely useless. F1 and ROC-AUC are the meaningful metrics here.

`scale_pos_weight=4` was used in XGBoost to handle class imbalance (ratio of negatives to positives ≈ 4:1).

---

## 📈 Key EDA Findings

- **Best time to fly:** 5–6 AM departures have only ~7% delay rate
- **Worst time to fly:** 6–9 PM departures hit ~27% delay rate — cascading delays from the day
- **Best month:** September (~15% delay rate) — post-summer, pre-winter
- **Worst month:** June (~23%) — peak thunderstorm season + summer travel
- **Most reliable airline:** Hawaiian Airlines (9.9% delay rate)
- **Worst airport:** Chicago O'Hare — ORD (24.5% delay rate among top 20 busiest)
- **#1 delay cause:** Late Aircraft — incoming plane delays cascade into next departure

---

## 🖥 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Overview | KPI cards, delay by month/hour/day charts |
| 🔮 Delay Predictor | Real-time XGBoost prediction with auto-filled route distance |
| ✈ Airline Analytics | Full carrier rankings, reliability scores, delay comparison |
| 🗺 Airport Intelligence | Delay hotspots, flight volume, riskiest/safest routes |
| ⚠ Delay Cause Analysis | Cause breakdown by type, airline, and season |

---

## 🚧 Roadmap

- [ ] SHAP explainability page — feature importance per prediction
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] LightGBM comparison
- [ ] `src/` production pipeline scripts

---

## 🛠 Tech Stack

```
Python 3.11      pandas · numpy · scikit-learn
XGBoost          gradient boosted classifier
Streamlit        interactive dashboard
Pyarrow          parquet data format
BTS Dataset      Bureau of Transportation Statistics
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/VedK5643/flight-delay-intelligence-system.git
cd flight-delay-intelligence-system
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Raw data files are not tracked due to size. Download from [Kaggle](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018) and run `notebooks/02_preprocessing.ipynb` to regenerate processed data.

---

## 👤 Author

**Vedagya** · BTech CSE  
GitHub: [VedK5643](https://github.com/VedK5643)