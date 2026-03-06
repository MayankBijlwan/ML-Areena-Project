# IEEE SB GEHU: ML Challenge - Binary Fault Detection

## 📌 Project Overview
This project was developed for the online qualifiers of the **ML Challenge by IEEE SB, GEHU**. The goal is to perform binary classification on device monitoring data to distinguish between normal operating conditions and faulty states.

### Problem Statement
Given 47 numerical features ($F01$–$F47$) captured by an embedded detection system, the task is to predict the `Class` of the device:
* **0**: Normal Operating Condition
* **1**: Faulty Condition

---

## 📊 Dataset Description
The dataset consists of quantitative measurements reflecting the internal state and environmental interactions of a device.
* **Training Data:** 43,776 rows with 47 features and 1 target variable (`Class`).
* **Test Data:** 10,944 rows with 47 features and an `ID` column for submission tracking.

---

## 🛠️ Technical Workflow

### 1. Exploratory Data Analysis (EDA)
* **Handling Imbalance:** Identified the class distribution to ensure the model doesn't overfit the majority class.
* **Feature Analysis:** Visualized feature correlations and distributions (specifically $F01$, $F28$, and $F33$) to understand non-linear relationships.

### 2. Data Preprocessing
* **Feature Scaling:** Applied `StandardScaler` to normalize the 47 numerical features, ensuring that features with larger magnitudes do not dominate the model training.
* **Data Splitting:** Used an 80-20 train-test split with `StratifiedKFold` to maintain class ratios during cross-validation.

### 3. Model Selection
I evaluated multiple models, including **Logistic Regression** and **Balanced Random Forest**, but selected **XGBoost** for the final submission.
* **Algorithm:** XGBoost Classifier.
* **Optimization:** Handled class imbalance using `scale_pos_weight`.
* **Performance:** Achieved an accuracy of **~98.4%** and an F1-score of **0.98** on the validation set.

---

## 🚀 How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost
