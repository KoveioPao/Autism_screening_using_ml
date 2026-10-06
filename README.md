# Autism Spectrum Disorder (ASD) Screening Prediction using Machine Learning

An end-to-end machine learning classification pipeline built to predict Autism Spectrum Disorder (ASD) screening outcomes based on behavioral traits and demographic factors.

## Dataset Profile
* **Total Samples:** 800 observations
* **Target Variable:** `Class/ASD` (Binary Classification: 0 = No ASD, 1 = ASD)
* **Pre-processing:** Addressed missing data by grouping sparse categorical rows into clean categories, pruned unbeneficial structural fields (`ID`, `age_desc`), and treated continuous column variations via the Interquartile Range (IQR) method to remove outliers.

## Project Workflow & Methodology
This project implements a standard machine learning workflow to clean, balance, and optimize multiple classification algorithms:
1. **Categorical Encoding:** Applied `LabelEncoder` to translate multi-category string attributes into numerical data.
2. **Class Imbalance Resolution:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** on the training set to artificially balance the minority class, ensuring the models learn effectively from both positive and negative cases.
3. **Cross-Validation Baseline:** Evaluated 6 diverse unoptimized algorithms (Decision Tree, Random Forest, XGBoost, Logistic Regression, SVM, and Naive Bayes) using **5-Fold Cross-Validation** to guarantee stable, reliable training benchmarks.
4. **Hyperparameter Optimization:** Utilized **`RandomizedSearchCV`** across 20 iterations per model to fine-tune the hyperparameter matrices.
5. **Model Selection & Serialization:** The **Support Vector Machine (SVC)** emerged as the champion classifier, achieving a training cross-validation accuracy of **92%**. The finalized model and its calibrated categorical parameters were saved as `best_model.pkl` and `encoders.pkl` via the `pickle` library.

## Final Model Evaluation (Test Set)
The champion SVM model was evaluated on an untouched, out-of-sample holdout test set (20% of the data) to measure generalized real-world performance:

* **Overall Test Accuracy:** 80.63%
* **Minority Class Recall (ASD Catch Rate):** 50.00% (An increase from the initial 36% baseline)
* **Confusion Matrix:**
  * True Negatives (Correctly predicted No ASD): 113
  * True Positives (Correctly predicted ASD): 16
  * False Positives (Healthy predicted as ASD): 15
  * False Negatives (ASD missed by the model): 16

## How to Run the Notebook
1. Click the **"Open in Colab"** badge at the top of the notebook file in this repository.
2. Ensure you upload the screening dataset into your active session storage.
3. Run all cells sequentially to train and evaluate the models.

