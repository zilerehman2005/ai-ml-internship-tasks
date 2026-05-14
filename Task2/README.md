# 📈 TSLA Stock Price Prediction — ML Project

Predict Tesla's next-day closing price using historical OHLCV data from Yahoo Finance, comparing **Linear Regression** and **Random Forest Regressor** models.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Features Used](#-features-used)
- [Model Details](#-model-details)
- [Notebook Walkthrough](#-notebook-walkthrough)
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
|-------|---------|----------|---------|
| **Linear Regression** ✅ | **6.72** | **9.74** | **0.9771** |
| Random Forest | 8.50 | 13.44 | 0.9563 |

**🏆 Best Model: Linear Regression** — achieves 97.7% variance explained with an average prediction error of just ~$6.72 per day.

---

## 📁 Project Structure

```
TSLA_Stock_Price_Prediction/
├── TSLA_Stock_Price_Prediction.ipynb   # Main notebook
└── README.md                           # This file
```

---

## 🛠 Setup & Installation

**Requirements:** Python 3.8+ · Jupyter Notebook or Google Colab

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn
```

Or use the notebook's built-in install cell (Cell 2):

```python
!pip install -q yfinance
```

---

## ▶️ How to Run

**Option A — Google Colab (Recommended)**
1. Upload `TSLA_Stock_Price_Prediction.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Click **Runtime → Run all**
3. All dependencies install automatically

**Option B — Local Jupyter**
```bash
pip install jupyter yfinance pandas numpy scikit-learn matplotlib seaborn
jupyter notebook TSLA_Stock_Price_Prediction.ipynb
```

---

## 🔢 Features Used

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `Open` | Opening price of the day | Reflects market sentiment at the opening bell |
| `High` | Intraday highest price | Captures upward volatility |
| `Low` | Intraday lowest price | Captures downward volatility |
| `Volume` | Number of shares traded | High volume often precedes significant price moves |

**Target variable:** `Close` price of the *next* trading day — created by shifting the Close column back by one row using `shift(-1)`.

---

## 🤖 Model Details

### Linear Regression
- Simple parametric baseline model
- Features scaled with `StandardScaler` so all inputs are on the same range
- Learns a linear equation: `NextClose = w₁·Open + w₂·High + w₃·Low + w₄·Volume + bias`
- Fast, interpretable, requires no hyperparameter tuning

### Random Forest Regressor
- Ensemble of 100 decision trees, max depth 10
- Tree-based models are scale-invariant, so raw features are used directly
- Feature importances: High (54%) > Low (37%) > Open (9%) > Volume (1%)
- More robust to outliers and captures non-linear patterns

### Train / Test Split
- **80% training** — earliest chronological data
- **20% testing** — most recent data
- `shuffle=False` — preserves time order, which is critical for financial data

---

## 📓 Notebook Walkthrough

The notebook follows a complete end-to-end ML pipeline across 10 sections:

**1. Problem Statement**
Defines the goal: given today's Open, High, Low, and Volume, predict tomorrow's closing price.

**2. Import Libraries**
Loads pandas, numpy, scikit-learn, matplotlib, seaborn, and yfinance.

**3. Download Dataset**
Fetches 5+ years of TSLA daily OHLCV data from Yahoo Finance using `yf.Ticker("TSLA").history(...)`.

**4. Data Exploration**
Inspects dataset shape (1,509 rows × 7 columns), checks data types, confirms no missing values, and reviews descriptive statistics (mean close ~$180, max ~$480).

**5. Data Visualization**
- **Closing price trend chart** — shows Tesla's dramatic price appreciation, especially the 2020–2021 peak
- **Correlation heatmap** — confirms Open, High, Low, Close are all >0.99 correlated; Volume is a more independent signal

**6. Feature Engineering**
Selects the four input features, creates the next-day target via `shift(-1)`, drops the one resulting NaN row, then splits into 80/20 train-test sets maintaining chronological order.

**7. Model Training**
Trains both Linear Regression and Random Forest Regressor on the training set and generates predictions on the held-out test set.

**8. Model Evaluation**
Computes MAE, RMSE, and R² for both models. Linear Regression outperforms Random Forest across all three metrics.

**9. Results Visualization**
- **Scatter plots** — actual vs predicted prices for each model, with a perfect-prediction reference line and R² annotation
- **Time series plot** — overlays both models' predictions on actual prices across the test period
- **Residual histograms** — shows prediction error distributions, both centered near zero

**10. Conclusion**
Summarises findings and outlines potential improvements such as technical indicators, longer forecasting horizons, and market context features.

**Bonus Cell**
Uses the trained models on the most recent available trading day to predict TSLA's next-day closing price, showing both the predicted value and the percentage change from today's close.

---

## ⚠️ Limitations & Future Work

| Limitation | Potential Improvement |
|-----------|----------------------|
| Only 4 basic price features | Add technical indicators: RSI, MACD, Bollinger Bands, moving averages |
| Using same-day High/Low to predict next-day Close | Shift features by 1 day to avoid any look-ahead |
| Single stock analysis | Expand to portfolio-level or sector ETF prediction |
| Next-day prediction only | Multi-step forecasting (3-day, 1-week, 1-month) |
| No market context | Incorporate S&P 500 / NASDAQ as exogenous features |
| Static trained model | Implement rolling-window retraining |
| No sentiment signal | Integrate news or social media sentiment data |

---

## 🧪 Technologies

| Library | Purpose |
|---------|---------|
| `yfinance` | Download historical stock data from Yahoo Finance |
| `pandas` | Data loading, cleaning, and time-series handling |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, feature scaling, metrics, train/test split |
| `matplotlib` | Price trend and prediction plots |
| `seaborn` | Correlation heatmap and styled visualizations |

---

## ⚠️ Disclaimer

> This project is for **educational and demonstration purposes only.**
> Stock price predictions carry inherent uncertainty. Past performance does not guarantee future results.
> **Do not use this model for actual trading or investment decisions.**
