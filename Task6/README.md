# 🏠 House Price Prediction Using Machine Learning

## 📌 Task Objective

The goal of this project is to build a robust regression-based machine learning system that accurately predicts house prices based on property features such as location, size, condition, and amenities.

**Key Objectives:**
- Perform comprehensive Exploratory Data Analysis (EDA) to understand price drivers
- Engineer new features to improve model performance
- Train and compare multiple regression models
- Build a reusable prediction system for new property listings
- Provide actionable insights for real estate stakeholders

**Real-World Business Use Cases:**
- **For Sellers:** Price their property optimally to sell faster
- **For Buyers:** Avoid overpaying by verifying market value
- **For Investors:** Identify profitable investment opportunities
- **For Banks:** Automate property valuation for loan approvals

---

## 📊 Dataset Used

| Property | Value |
|----------|-------|
| **Source** | Real Estate Dataset (`data.csv`) |
| **Shape** | 4,600 rows × 18 columns |
| **Target Variable** | `price` (House price in USD) |
| **Feature Types** | Numerical & Categorical |

### Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| `date` | Date | Sale date of the property |
| `price` | Numeric | **Target variable** — Sale price in USD |
| `bedrooms` | Integer | Number of bedrooms |
| `bathrooms` | Float | Number of bathrooms |
| `sqft_living` | Integer | Living area in square feet |
| `sqft_lot` | Integer | Lot size in square feet |
| `floors` | Float | Number of floors |
| `waterfront` | Binary | Waterfront access (0=No, 1=Yes) |
| `view` | Integer | View quality rating (0–4) |
| `condition` | Integer | Condition rating (1–5) |
| `sqft_above` | Integer | Above-ground square footage |
| `sqft_basement` | Integer | Basement square footage |
| `yr_built` | Integer | Year the house was built |
| `yr_renovated` | Integer | Year of renovation (0 if never) |
| `street` | Categorical | Street address |
| `city` | Categorical | City name |
| `statezip` | Categorical | State and ZIP code |
| `country` | Categorical | Country (constant: USA) |

### Data Preprocessing Steps
- Extracted `sale_year` and `sale_month` from the `date` column
- Dropped constant `country` column (single unique value)
- Capped outliers in `price` using the IQR method
- Label Encoded high-cardinality categorical features (`city`, `street`, `statezip`)
- Applied StandardScaler for feature normalization

---

## 🤖 Models Applied

Four regression models were trained and evaluated on the dataset:

| # | Model | Description |
|---|-------|-------------|
| 1 | **Linear Regression** | Baseline model — fast and interpretable |
| 2 | **Random Forest Regressor** | Ensemble of decision trees; handles non-linearity well |
| 3 | **Gradient Boosting Regressor** | Sequential error correction; high accuracy potential |
| 4 | **XGBoost Regressor** | Optimized gradient boosting; often wins ML competitions |

### Model Selection Rationale
- **Linear Regression** serves as a simple, interpretable baseline
- **Random Forest** captures complex feature interactions through bagging
- **Gradient Boosting** sequentially corrects residuals for improved accuracy
- **XGBoost** provides state-of-the-art performance with regularization

### Evaluation Metrics
- **MAE** (Mean Absolute Error): Average magnitude of prediction errors
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily
- **R² Score**: Proportion of variance explained by the model (1.0 = perfect)

---

## 🔑 Key Results and Findings

### Best Performing Model
The **XGBoost Regressor** emerged as the best-performing model, achieving the highest R² Score among all candidates.

### Key Insights from EDA
- **Square footage (`sqft_living`)** is the strongest predictor of house price
- **Location (`city`, `statezip`)** significantly impacts property valuation
- **Waterfront access** commands a substantial price premium
- **House condition** correlates positively with sale price
- **House age and renovation status** are critical value factors

### Feature Engineering Impact
Three engineered features were created to boost model performance:
- **`house_age`** = Current year − `yr_built` — captures depreciation/appreciation
- **`is_renovated`** = Binary flag (`yr_renovated` > 0) — renovation premium
- **`total_sqft`** = `sqft_above` + `sqft_basement` — aggregate living space

### Challenges Addressed
- High cardinality in categorical variables (street names) required Label Encoding
- Outliers in price distribution were capped using IQR bounds
- Feature engineering was necessary to capture implicit relationships

### Limitations
- Model relies on historical data and may not capture sudden market shifts
- Label Encoding assumes ordinal relationships for categorical features
- External factors (economic downturns, interest rates) are not included

### Future Improvements
- Implement geospatial clustering for neighborhood effects
- Use Target Encoding for high-cardinality categorical features
- Hyperparameter tuning with GridSearchCV or Optuna
- Deploy as a REST API using Flask or FastAPI

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `House_Price_Prediction.ipynb`
3. Upload `data.csv` (or let the script generate synthetic data)
4. Run all cells sequentially (`Runtime` → `Run all`)

### Option 2: Local Environment
```bash
# Clone the repository
git clone <repo-url>
cd house-price-prediction-ml

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/House_Price_Prediction.ipynb
```

---

## 📁 Repository Structure

```
house-price-prediction-ml/
│
└── data.csv
│
└── House_Price_Prediction.ipynb
├── 📄 README.md
├── 📄 requirements.txt
```

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — Machine learning pipelines
- **XGBoost** — Advanced gradient boosting

---

## 📝 Requirements

```
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==1.7.6
```


## 📄 License

This project is open-source and available for educational and personal use.
