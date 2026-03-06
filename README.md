# 🛡️ Embedded System Fault Detection 
### ML Challenge | IEEE SB, GEHU (Online Qualifiers)

## 📌 Project Overview
This repository contains a high-performance Machine Learning solution designed to classify the operational status of hardware devices. Using 47 distinct quantitative sensor measurements ($F01$–$F47$), the model identifies whether a device is functioning under **Normal (Class 0)** conditions or exhibiting a **Faulty (Class 1)** state.

The project addresses the critical need for automated anomaly detection in embedded monitoring systems to prevent hardware failure and optimize maintenance cycles.



## 📊 Dataset Description
The dataset consists of measurements captured by an embedded detector during device activity cycles.
* **Input Features:** 47 Numerical features representing internal states and environmental interactions.
* **Target Variable:** * `0`: Normal
    * `1`: Faulty
* **Data Split:** * `TRAIN.csv`: Used for training and cross-validation.
    * `TEST.csv`: Used for generating final blind predictions.

## 🛠️ Technical Pipeline

### 1. Exploratory Data Analysis (EDA)
* **Feature Profiling:** Conducted statistical analysis on features $F01$ through $F47$ to identify variance and distribution patterns.
* **Correlation Analysis:** Identified key sensor metrics that show high sensitivity to device faults.
* **Data Integrity:** Confirmed zero missing values across the 43,776 training samples.

### 2. Preprocessing & Engineering
* **Feature Scaling:** Implemented `StandardScaler` to bring all 47 features onto a uniform scale. This is crucial for models like Logistic Regression to ensure features with larger magnitudes don't dominate the objective function.
* **Stratification:** Used stratified sampling to maintain the class ratio across training and validation sets, ensuring the model is equally robust at detecting faults as it is at identifying normal behavior.

### 3. Model Architecture
We utilized an optimized **Logistic Regression** framework, fine-tuned through **GridSearchCV**.
* **Hyperparameter Tuning:** Optimized the regularization strength (`C`) and solvers (`lbfgs`) to prevent overfitting.
* **Validation:** Employed 5-Fold Cross-Validation to ensure the model generalizes well to unseen sensor data.



## 📈 Performance Metrics
The model was evaluated primarily on its ability to minimize "False Negatives" (missing a fault), while maintaining high overall accuracy.
* **Accuracy:** ~74%
* **Cross-Validation F1-Score:** ~0.654
* **Robustness:** The model demonstrated consistent performance across different data folds, indicating high reliability for real-world embedded deployment.

## 🚀 How to Run
1.  **Environment Setup:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
2.  **Training:** Run the `IEEEML.ipynb` notebook to preprocess data and train the model.
3.  **Inference:** The model automatically processes `TEST.csv` and exports predictions.
4.  **Submission:** The final output is saved as `FINAL.csv` with the required `ID` and `CLASS` columns.

## 📁 Repository Structure
* `IEEEML.ipynb`: Full end-to-end data science pipeline.
* `FINAL.csv`: Final prediction output for evaluation.
* `README.md`: Project documentation.

---
**Developed for the ML Challenge by IEEE SB, GEHU.**
*Note: This project is for educational purposes.*
