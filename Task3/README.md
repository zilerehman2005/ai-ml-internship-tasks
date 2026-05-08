# Heart Disease Prediction - ML Classification Project

## Task 3: Heart Disease Prediction (AI/ML Internship)

**Created on:** Google Colab  
**Dataset:** `heart_disease_uci.csv` (UCI Heart Disease Dataset - 920 samples)  
**Environment:** Python 3, Google Colab Notebook

---

## Objective

Build a machine learning model to predict whether a person is at risk of heart disease based on their health data using the UCI Heart Disease Dataset.

---

## Dataset Information

- **Source:** UCI Heart Disease Dataset (Combined from 4 locations: Cleveland, Hungary, Switzerland, VA Long Beach)
- **File Name:** `heart_disease_uci.csv`
- **Size:** 920 patients
- **Features:** 16 columns (including `id` and target `num`)
- **Target Variable:** `num` (0 = No Disease, 1-4 = Disease Severity)

### Feature Descriptions

| Column | Non-Null Count | Dtype | Description |
|--------|---------------|-------|-------------|
| id | 920 | int64 | Patient ID |
| age | 920 | int64 | Age in years |
| sex | 920 | object | Sex (Male/Female) |
| dataset | 920 | object | Data source location (hospital) |
| cp | 920 | object | Chest Pain Type |
| trestbps | 861 | float64 | Resting Blood Pressure (mm Hg) |
| chol | 890 | float64 | Serum Cholesterol (mg/dl) |
| fbs | 830 | object | Fasting Blood Sugar > 120 mg/dl (True/False) |
| restecg | 918 | object | Resting ECG Results |
| thalch | 865 | float64 | Maximum Heart Rate Achieved |
| exang | 865 | object | Exercise Induced Angina (True/False) |
| oldpeak | 858 | float64 | ST Depression Induced by Exercise |
| slope | 611 | object | Slope of Peak Exercise ST Segment |
| ca | 309 | float64 | Number of Major Vessels Colored by Fluoroscopy (0-3) |
| thal | 434 | object | Thalassemia (Normal/Fixed/Reversible Defect) |
| num | 920 | int64 | Target: Heart Disease Severity (0-4) |

### Missing Values Summary

| Column | Missing Count | Missing % | Handling Strategy |
|--------|--------------|-----------|-------------------|
| trestbps | 59 | 6.4% | Filled with median |
| chol | 30 | 3.3% | Filled with median |
| fbs | 90 | 9.8% | Filled with mode |
| restecg | 2 | 0.2% | Filled with mode |
| thalch | 55 | 6.0% | Filled with median |
| exang | 55 | 6.0% | Filled with mode |
| oldpeak | 62 | 6.7% | Filled with median |
| slope | 309 | 33.6% | Filled with mode |
| ca | 611 | 66.4% | Filled with median |
| thal | 486 | 52.8% | Filled with mode |

---

## Preprocessing Steps

1. **Dropped `id` column** - Not useful for prediction
2. **Converted target `num` to binary**:
   - `0` = No Heart Disease
   - `1, 2, 3, 4` = Heart Disease (combined into single class)
3. **Handled missing values**:
   - Numerical columns (`trestbps`, `chol`, `thalch`, `oldpeak`, `ca`): Filled with **median**
   - Categorical columns (`fbs`, `restecg`, `exang`, `slope`, `thal`): Filled with **mode**
4. **Label encoded categorical variables**: `sex`, `dataset`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`
5. **Applied StandardScaler** for Logistic Regression
6. **Train-Test Split**: 80% train, 20% test with stratification

---

## Exploratory Data Analysis (EDA)

Performed comprehensive EDA including:
- Target distribution analysis
- Age distribution by heart disease status
- Sex vs heart disease correlation
- Chest pain type analysis
- Cholesterol, blood pressure, and heart rate distributions
- ST depression (oldpeak) analysis
- Exercise angina correlation
- Feature correlation heatmap

**Key EDA Findings:**
- Dataset is approximately balanced (~50% disease, ~50% no disease)
- Chest pain type shows strong correlation with heart disease
- Maximum heart rate (`thalach`) is lower in diseased patients
- ST depression (`oldpeak`) is higher in diseased patients
- Data source (`dataset`) shows variation across hospitals

---

## Models Applied

### 1. Logistic Regression
- **Hyperparameter Tuning:** GridSearchCV with `C` [0.001, 0.01, 0.1, 1, 10, 100], `penalty` ['l1', 'l2']
- **Scaling:** StandardScaler applied
- **Best Parameters:** Determined via 5-fold cross-validation

### 2. Decision Tree
- **Hyperparameter Tuning:** GridSearchCV with `max_depth` [3, 5, 7, 10, None], `min_samples_split` [2, 5, 10], `criterion` ['gini', 'entropy']
- **Best Parameters:** Determined via 5-fold cross-validation

### 3. Random Forest (Ensemble)
- **Hyperparameter Tuning:** GridSearchCV with `n_estimators` [100, 200], `max_depth` [5, 10, None]
- **Best Parameters:** Determined via 5-fold cross-validation

---

## Evaluation Metrics

- **Accuracy:** Overall correctness of predictions
- **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve
- **Confusion Matrix:** TP, TN, FP, FN breakdown
- **Precision:** Accuracy of positive predictions
- **Recall (Sensitivity):** Coverage of actual positive cases
- **F1-Score:** Harmonic mean of precision and recall
- **5-Fold Cross-Validation:** For model stability assessment

---

## Results

### Test Set Performance

| Model | Accuracy | ROC-AUC | Precision (Disease) | Recall (Disease) | F1 (Disease) |
|-------|----------|---------|--------------------|--------------------|--------------|
| **Random Forest** | **82.07%** | **0.930** | 0.805 | **0.892** | **0.847** |
| Logistic Regression | 81.52% | 0.907 | 0.815 | 0.863 | 0.838 |
| Decision Tree | 80.98% | 0.887 | **0.853** | 0.794 | 0.822 |

### Confusion Matrices

**Logistic Regression:**
```
                Predicted
              No    Yes
Actual No     62     20
Actual Yes    14     88
```

**Decision Tree:**
```
                Predicted
              No    Yes
Actual No     68     14
Actual Yes    21     81
```

**Random Forest:**
```
                Predicted
              No    Yes
Actual No     60     22
Actual Yes    11     91
```

### 5-Fold Cross-Validation Results

| Model | Mean ROC-AUC | Std Dev | Stability |
|-------|-------------|---------|-----------|
| Logistic Regression | ~0.909 | ±0.044 | Stable |
| Decision Tree | ~0.830 | ±0.076 | Moderate |
| Random Forest | ~0.909 | ±0.047 | Stable |

---

## Feature Importance Analysis

### Top 5 Most Important Features (by Model)

**Logistic Regression (by |coefficient|):**
1. `ca` (Number of Major Vessels): 0.540 - **increases risk**
2. `exang` (Exercise Angina): 0.513 - **increases risk**
3. `cp` (Chest Pain Type): -0.483 - **decreases risk** (negative correlation with encoded value)
4. `oldpeak` (ST Depression): 0.447 - **increases risk**
5. `dataset` (Data Source): 0.438 - **increases risk**

**Decision Tree:**
1. `cp` (Chest Pain Type): 0.568
2. `dataset` (Data Source): 0.242
3. `exang` (Exercise Angina): 0.091
4. `age`: 0.078
5. `sex`: 0.021

**Random Forest:**
1. `cp` (Chest Pain Type): 0.182
2. `exang` (Exercise Angina): 0.156
3. `dataset` (Data Source): 0.124
4. `oldpeak` (ST Depression): 0.105
5. `thalch` (Max Heart Rate): 0.096

### Medical Interpretation

- **Chest Pain Type (`cp`)**: Most predictive feature across all models. Atypical angina strongly correlates with heart disease.
- **Exercise Angina (`exang`)**: Classic symptom - patients experiencing angina during exercise are at higher risk.
- **Number of Major Vessels (`ca`)**: Direct measure of coronary artery blockage. More blocked vessels = higher risk.
- **ST Depression (`oldpeak`)**: ECG marker of ischemia during exercise. Higher values indicate reduced blood flow.
- **Data Source (`dataset`)**: Indicates variation across hospitals. Different locations may have different patient populations or diagnostic practices.
- **Max Heart Rate (`thalach`)**: Lower maximum heart rate suggests reduced cardiac reserve capacity.

---

## Best Model Recommendation

**Random Forest** is recommended as the primary model:
- Highest ROC-AUC: **0.930** (excellent discrimination)
- Highest Recall for Disease: **89.2%** (catches most cases)
- Good balance between precision and recall
- Robust feature importance analysis
- Stable cross-validation performance

---

## Key Insights & Clinical Recommendations

1. **Chest pain evaluation** should be the primary focus in risk assessment
2. **Exercise stress tests** are critical - monitor ST depression and angina symptoms
3. **Coronary angiography** (number of blocked vessels) is a direct risk indicator
4. **Maximum heart rate** during stress tests indicates cardiac fitness
5. **Data source variation** suggests different hospitals may need calibrated thresholds
6. Always combine model predictions with clinical judgment

---

## How to Run This Project

### On Google Colab (Original Environment)
1. Upload `heart_disease_uci.csv` to your Google Colab session
2. Run all cells in the notebook sequentially
3. Visualizations will display inline
4. Download the `.ipynb` file for GitHub

### Local Environment
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook heart_disease_prediction.ipynb
```

---

## Project Files

```
heart-disease-prediction/
├── heart_disease_prediction.ipynb    # Main Google Colab notebook
├── heart_disease_uci.csv             # Dataset (not pushed to GitHub if large)
├── README.md                         # This file
└── outputs/                          # Generated visualizations
    ├── eda_visualizations.png
    ├── correlation_heatmap.png
    ├── roc_confusion_matrices.png
    └── feature_importance.png
```

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

*(All pre-installed in Google Colab)*

---

## Limitations & Future Work

1. **Missing Data:** Some features (`ca`: 66.4%, `thal`: 52.8%) have significant missing values. Consider advanced imputation techniques (KNN, MICE) or collecting more complete data.

2. **Data Source Bias:** The `dataset` feature (hospital location) appears as an important predictor, suggesting potential population or diagnostic differences across centers.

3. **Class Imbalance:** While approximately balanced, slight imbalance exists. SMOTE or class weighting could be explored.

4. **Feature Engineering:** Could create interaction features (e.g., age × cholesterol, BP × heart rate).

5. **Model Ensembling:** Could combine all three models for improved performance.

6. **External Validation:** Model should be tested on completely unseen data from different hospitals.

---

## Skills Demonstrated

- Binary Classification
- Medical Data Understanding & Interpretation
- Handling Missing Data (up to 66% in some columns)
- Categorical Variable Encoding (Label Encoding)
- Model Evaluation (Accuracy, ROC-AUC, Confusion Matrix, Precision, Recall, F1)
- Feature Importance Analysis
- Cross-Validation & Hyperparameter Tuning (GridSearchCV)
- Data Visualization & EDA
- Clinical Insight Extraction

---

## References

- UCI Machine Learning Repository: Heart Disease Dataset
- Dataset sources: Cleveland Clinic Foundation, Hungarian Institute of Cardiology, V.A. Medical Center Long Beach, University Hospital Zurich

---

## Author

AI/ML Internship Task Submission  
Created on Google Colab

---

*Last Updated: May 2026*
