# 🛡️ Embedded System Fault Detection 
### **ML Challenge | IEEE SB, GEHU (Online Qualifiers)**

## 📌 Project Overview
This repository contains a robust Machine Learning pipeline developed for the IEEE ML Challenge. The objective is to monitor embedded devices and classify their operational status based on 47 quantitative sensor metrics ($F01$–$F47$). 

The system distinguishes between:
* **Class 0:** Normal Operating Conditions
* **Class 1:** Faulty Condition (Anomaly Detected)

## 📊 Data Intelligence & EDA
The dataset consists of **43,776 training samples**. Our Exploratory Data Analysis revealed:
* **Data Integrity:** 0 missing values across all features.
* **High Variance:** Features like $F31$ and $F37$ showed significantly higher scales, requiring **Standardization**.
* **Class Distribution:** A relatively balanced dataset (~60/40 split), which we further addressed using balanced class weights in our models.
* **Non-Linearity:** Visualizations indicated that the boundary between Normal and Faulty states is non-linear, suggesting that tree-based ensembles would outperform linear models.



## ⚙️ The Technical Pipeline

### 1. Preprocessing
* **Feature Scaling:** We utilized `StandardScaler` to normalize the 47 features. This ensures that the model treats all sensor inputs with equal mathematical importance.
* **Data Splitting:** A stratified 80/20 split was used to ensure the validation set perfectly mirrors the class distribution of the training data.

### 2. Model Evolution & Experimentation
We didn't just pick one model; we experimented with multiple architectures to find the most reliable solution:

| Model | Why we chose it | Performance/Observation |
| :--- | :--- | :--- |
| **Logistic Regression** | To establish a linear baseline. | **74% Accuracy**. Confirmed that the data has complex non-linear patterns. |
| **Random Forest** | To capture non-linear interactions between sensors. | **98.05% Accuracy**. Massive improvement; handled the high-dimensional data exceptionally well. |
| **XGBoost (Final)** | Industry-standard for tabular data; uses gradient boosting to minimize residual errors. | **98.42% Accuracy**. Best performing model with superior F1-Score for the "Faulty" class. |

### 3. Hyperparameter Optimization
We used **GridSearchCV** with **5-Fold Stratified Cross-Validation** to fine-tune our models. For the baseline, we optimized:
* `C` (Regularization strength): Tested `[0.01, 0.1, 1, 10, 100]`
* `Solvers`: Evaluated `lbfgs` vs `liblinear` vs `saga`.



## 📈 Final Results (XGBoost)
The final model achieved a near-perfect classification on the validation set:
* **Precision:** 0.98
* **Recall:** 0.98
* **F1-Score:** 0.98
* **Validation Accuracy:** **98.42%**

## 🚀 How to Use
1.  **Dependencies:** Ensure you have `pandas`, `scikit-learn`, and `xgboost` installed.
2.  **Run Notebook:** Execute `IEEEML.ipynb`. It will:
    * Load the data directly from the GitHub source.
    * Preprocess and Scale the features.
    * Train the optimized XGBoost model.
    * Generate predictions for `TEST.csv`.
3.  **Submission:** The script outputs a `FINAL.csv` formatted exactly as per competition requirements (`ID` -> `Prediction`).

## 📁 File Structure
* `IEEEML.ipynb`: The complete end-to-end research and modeling notebook.
* `FINAL.csv`: The generated predictions for the 10,944 test entries.
* `TRAIN.csv` & `TEST.csv`: Dataset files provided by IEEE SB, GEHU.

---
**Event:** ML CHALLENGE by IEEE SB, GEHU  
*Note: This model is built for educational and competition purposes.*
