# Hotel Cancellation Analysis
**Course: IEE 520 - Statistical Learning for Data Mining**

**Institution: Arizona State University**

**Semester: Fall 2025**

## Project Overview
This project implements a binary classification pipeline optimized for Balanced Error Rate (BER) using XGBoost. The challenge involved classifying 10,000+ samples with mixed numerical and categorical features, handling class imbalance, and tuning hyperparameters to minimize prediction error across both classes.

## The Challenge
Standard accuracy metrics can be misleading with imbalanced datasets—a model predicting all samples as the majority class achieves high accuracy but fails on the minority class. BER addresses this by weighting errors equally across classes:

## My Contributions
**1. Model Selection & Architecture**

I evaluated multiple ensemble methods and selected XGBoost as the optimal classifier based on 5-fold stratified cross-validation performance.

**Why XGBoost:**
- Native handling of mixed data types
- Built-in regularization (L1/L2) prevents overfitting
- scale_pos_weight parameter directly addresses class imbalance
- Efficient parallel processing
- XGBoost achieved the best BER score in the least allotted run-time

**2. Hyperparameter Tuning**

I implemented a comprehensive hyperparameter optimization strategy using RandomizedSearchCV.

**Parameter Space:**
- n_estimators: randint(100, 600)
- max_depth: randint(3, 15)
- learning_rate: Uniform(0.01, 0.29)
- subsample: Uniform(0.5, 0.5)
- colsample_bytree: Uniform(0.5, 0.5)
- min_child_weight: randint(1, 15)
- gamma: Uniform(0, 1)
- reg_alpha: Uniform(0, 2)
- reg_lambda: Uniform(0.5, 2)

**Tuning Strategy:**
- 100 random combinations tested
- 5-fold stratified cross-validation per combination
- Scoring metric: Balanced Accuracy (1 - BER)
- Total evaluations: 500 model fits

**3. Pre-Processing Pipeline:**

I designed a robust preprocessing pipeline to handle mixed data types and missing values.

**Numerical Features:**
- StandardScaler normalization
- Median imputation for missing values

**Categorical Features:**
- OneHotEncoder with handle_unknown='ignore'
- Mode imputation for missing values

**Class Imbalance Handling:**
- Calculated scale_pos_weight = count(negative) / count(positive)
- Applied during model training to penalize minority class errors

**4. Model Evaluation & Validation:**

I implemented rigorous evaluation using stratified cross-validation to ensure reliable BER estimates. The model identified the most predictive features for classification.

**Evaluation Metrics:**
- Balanced Error Rate (BER): 0.34
- Balanced Accuracy: 0.66
- ROC-AUC: 0.72
- Confusion matrix analysis per class

**Validation Approach:**
- 5-fold Stratified K-Fold (preserves class distribution)
- Out-of-fold predictions for unbiased evaluation
- Final model trained on 100% of labeled data

**Results:**
- BER: 0.34
- Balanced Accuracy: 0.66
- Class 0 Error Rate: 0.32
- Class 1 Error Rate: 0.36
