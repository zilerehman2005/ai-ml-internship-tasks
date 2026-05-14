# 📈 TSLA Stock Price Prediction — ML Project

Predict Tesla's next-day closing price using historical OHLCV data from Yahoo Finance, comparing **Linear Regression** and **Random Forest Regressor** models.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Results](#-results)
- [Bugs Found & Fixed](#-bugs-found--fixed)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Features](#-features-used)
- [Model Details](#-model-details)
- [Limitations & Future Work](#-limitations--future-work)
- [Disclaimer](#-disclaimer)

---

## 🎯 Project Overview

| Property | Value |
|----------|-------|
| **Stock** | Tesla Inc. (TSLA) |
| **Data Source** | Yahoo Finance via `yfinance` |
| **Period** | 2019-01-01 → 2024-12-31 (~1,508 trading days) |
| **Task** | Predict **next-day closing price** (regression) |
| **Models** | Linear Regression · Random Forest Regressor |
| **Metrics** | MAE · RMSE · R² Score |

---

## 📊 Results

| Model | MAE ($) | RMSE ($) | R² Score |
|-------|---------|---------|---------|
| **Linear Regression** ✅ | **6.72** | **9.74** | **0.9771** |
| Random Forest | 8.50 | 13.44 | 0.9563 |

**🏆 Winner: Linear Regression** — achieves 97.7% variance explained with a lower average error of ~$6.72 per prediction.

> **Note:** The original notebook incorrectly stated Random Forest was the best model in the conclusion. This was a documentation error — the metrics table always showed Linear Regression winning.

---

## 🐛 Bugs Found & Fixed

### Bug 1 — Missing Feature Scaling (Critical)
**Location:** Cell 7.1 — Linear Regression Training

**Problem:**  
`Volume` values (~10⁸) are eight orders of magnitude larger than price columns (~$100–500). Without scaling, `LinearRegression` assigns a coefficient of `0.000000` to Volume, completely ignoring it.

```
# Original output — Volume effectively ignored:
• Open:   -0.606742
• High:    0.782202
• Low:     0.823433
• Volume:  0.000000   ← ❌ lost due to scale mismatch
```

**Fix:**  
Added `StandardScaler` to normalize all features to mean=0, std=1 before fitting Linear Regression.

```python
# FIXED — added in cell 3 (imports)
from sklearn.preprocessing import StandardScaler

# FIXED — added in cell 7.1 (LR training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
```

---

### Bug 2 — Scaler Not Applied in Bonus Prediction Cell (Critical)
**Location:** Cell 10 (Bonus — Predict Tomorrow)

**Problem:**  
The bonus prediction cell feeds raw (unscaled) features directly into the trained Linear Regression model, which was trained on scaled data. This produces an incorrect prediction.

```python
# ORIGINAL — wrong, raw features used with a model trained on scaled data
latest_data = stock_data.iloc[-1][feature_columns].values.reshape(1, -1)
tomorrow_prediction = rf_model.predict(latest_data)[0]  # also used rf, not lr
```

**Fix:**
```python
# FIXED — apply the same scaler before predicting with LR
latest_data_scaled = scaler.transform(latest_data_raw)
tomorrow_lr = lr_model.predict(latest_data_scaled)[0]
tomorrow_rf = rf_model.predict(latest_data_raw)[0]   # RF needs no scaling
```

---

### Bug 3 — Wrong Best-Model Claim in Conclusion (Documentation)
**Location:** Cell 10 (Conclusion markdown)

**Problem:**  
The conclusion hardcoded *"Best Model: Random Forest Regressor"* but the metrics clearly show Linear Regression wins on all three metrics.

**Fix:**  
Updated conclusion to correctly state Linear Regression is the better model, with explanation of why scaling enabled this result.

---

### Bug 4 — Overly Complex Missing-Value Alignment (Minor)
**Location:** Cell 6.2 — Handle Missing Values

**Problem:**  
Used `dropna()` separately on `X` and `y`, then re-intersected indices — unnecessary complexity that could silently drop extra rows if data has other NaNs.

```python
# ORIGINAL — fragile
X_clean = X.dropna()
y_clean = y.dropna()
common_index = X_clean.index.intersection(y_clean.index)
X_clean = X_clean.loc[common_index]
y_clean = y_clean.loc[common_index]
```

**Fix:**  
Since `shift(-1)` creates exactly 1 NaN at the last row, simply drop it directly:

```python
# FIXED — explicit and correct
X_clean = X.iloc[:-1].copy()
y_clean = y.iloc[:-1].copy()
```

---

## 📁 Project Structure

```
TSLA_Stock_Price_Prediction/
├── TSLA_Stock_Price_Prediction_fixed.ipynb   # Fixed notebook (run this)
├── TSLA_Stock_Price_Prediction.ipynb         # Original notebook (for reference)
└── README.md                                 # This file
```

---

## 🛠 Setup & Installation

### Requirements
- Python 3.8+
- Jupyter Notebook or Google Colab

### Install Dependencies

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn
```

Or inside the notebook (already included in Cell 2):

```python
!pip install -q yfinance
```

---

## ▶️ How to Run

### Option A — Google Colab (Recommended)
1. Upload `TSLA_Stock_Price_Prediction_fixed.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Click **Runtime → Run all**
3. All dependencies install automatically

### Option B — Local Jupyter
```bash
# Clone or download the notebook, then:
pip install jupyter yfinance pandas numpy scikit-learn matplotlib seaborn
jupyter notebook TSLA_Stock_Price_Prediction_fixed.ipynb
```

---

## 🔢 Features Used

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `Open` | Opening price of the day | Reflects market sentiment at the bell |
| `High` | Intraday highest price | Captures upward volatility |
| `Low` | Intraday lowest price | Captures downward volatility |
| `Volume` | Number of shares traded | High volume often precedes price moves |

**Target:** `Close` of the *next* trading day (created via `shift(-1)`)

---

## 🤖 Model Details

### Linear Regression
- Simple parametric baseline
- Features scaled with `StandardScaler` *(bug fix applied)*
- Learns: `NextClose = w₁·Open + w₂·High + w₃·Low + w₄·Volume + bias`
- Fast, interpretable, no hyperparameter tuning

### Random Forest Regressor
- 100 trees, max depth 10
- Uses raw (unscaled) features — tree models are scale-invariant
- Feature importances: High (54%) > Low (37%) > Open (9%) > Volume (1%)
- Robust to outliers, captures non-linear patterns

### Train/Test Split
- **80% training** (chronological — earliest data)
- **20% testing** (most recent data)
- `shuffle=False` — preserves time order (critical for financial data)

---

## ⚠️ Limitations & Future Work

| Limitation | Potential Improvement |
|-----------|----------------------|
| Only 4 basic price features | Add technical indicators: RSI, MACD, Bollinger Bands, moving averages |
| Potential data leakage: using same-day High/Low | Use *previous day's* OHLCV only (lag features) |
| Single stock | Expand to portfolio prediction or sector ETFs |
| Next-day only | Multi-step forecasting (3-day, 5-day, 1-week) |
| No market context | Add S&P 500 / NASDAQ as exogenous features |
| Static model | Retrain weekly/monthly with rolling window |
| No sentiment data | Integrate news sentiment or social media signals |

---

## 🧪 Technologies

| Library | Version | Purpose |
|---------|---------|---------|
| `yfinance` | ≥0.2 | Download historical stock data from Yahoo Finance |
| `pandas` | ≥1.3 | Data loading, cleaning, time-series handling |
| `numpy` | ≥1.21 | Numerical operations |
| `scikit-learn` | ≥1.0 | ML models, scaling, metrics, train/test split |
| `matplotlib` | ≥3.4 | Plotting price trends and predictions |
| `seaborn` | ≥0.11 | Correlation heatmaps and styled plots |

---

## ⚠️ Disclaimer

> This project is for **educational and demonstration purposes only**.  
> Stock price predictions carry inherent uncertainty. Past performance does not guarantee future results.  
> **Do not use this model for actual trading or investment decisions.**
