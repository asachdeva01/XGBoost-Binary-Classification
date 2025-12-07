import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, make_scorer, confusion_matrix
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['XGB_VERBOSITY'] = '0'
import xgboost as xgb

# Configuration
labeled_path = 'ProjectLABELED2025.csv'
unlabeled_path = 'ProjectNotLABELED2025.csv'
output_path = 'ProjectPredictions2025AbhiSachdeva.csv'
target_column = 'label'

# Data Overview
binary_columns = ['x1', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14']
categorical_columns = ['x2', 'x3', 'x4']
numerical_columns = ['x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21']
columns = binary_columns + categorical_columns + numerical_columns

# Settings
N = 100
cv_folds = 5
rand_state = 42

def loadData():

    # Load datasets
    labeled = pd.read_csv(labeled_path)
    unlabeled = pd.read_csv(unlabeled_path)

    # Find the index column
    index_column = unlabeled.columns[0]
    if index_column not in columns + [target_column]:
        unlabeled_indices = unlabeled[index_column].values
    else:
        unlabeled_indices = np.arange(len(unlabeled))
    
    feature_columns = [col for col in columns if col in labeled.columns]
    x_labeled = labeled[feature_columns].copy()
    y_labeled = labeled[target_column].copy()
    x_unlabeled = unlabeled[feature_columns].copy()

    # Imputate using median value
    for col in numerical_columns:
        if col in x_labeled.columns:
            median = x_labeled[col].median()
            x_labeled[col].fillna(median, inplace=True)
            x_unlabeled[col].fillna(median, inplace=True)
    
    # Imputate using mode value
    for col in binary_columns + categorical_columns:
        if col in x_labeled.columns:
            mode = x_labeled[col].mode()[0]
            x_labeled[col].fillna(mode, inplace=True)
            x_unlabeled[col].fillna(mode, inplace=True)
    
    class_counts = y_labeled.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    return x_labeled, y_labeled, x_unlabeled, unlabeled_indices, scale_pos_weight

def computeHyperparameters(x_labeled, y_labeled, scale_pos_weight):

    # Hyperparameter grid
    parameter_grid = {
        'n_estimators': randint(100, 350),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.02, 0.18),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'min_child_weight': randint(1, 8),
        'gamma': uniform(0, 0.3),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0.5, 1),  
    }

    # XGBoost classifier model
    base_model = xgb.XGBClassifier(
        scale_pos_weight = scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        verbosity=0,
        random_state=rand_state,
        n_jobs=-1 
    )

    # 5 fold stratified cross validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rand_state)
    scorer = make_scorer(balanced_accuracy_score)

    # Find optimal hyperparameters
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_grid,
        n_iter=N,
        scoring=scorer,
        cv=cv,
        random_state=rand_state,
        n_jobs=-1,
        verbose=0
    )

    search.fit(x_labeled, y_labeled)
    return search.best_estimator_, search.best_params_

def evaluateModel(model, x_labeled, y_labeled):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rand_state)
    y_predicted_cv = cross_val_predict(model, x_labeled, y_labeled, cv=cv, n_jobs=-1)

    # Generate a confusion matrix
    matrix = confusion_matrix(y_labeled, y_predicted_cv)

    classes = sorted(y_labeled.unique())
    error_rates = []

    # Calculate the error rates and the BER
    for cls in classes:
        mask = (y_labeled == cls)
        total = mask.sum()
        correct = ((y_labeled == cls) & (y_predicted_cv == cls)).sum()
        error_rate = (total - correct) / total
        error_rates.append(error_rate)

    ber = np.mean(error_rates)
    return ber, matrix, error_rates

def makePredictions(model, x_labeled, y_labeled, x_unlabeled, indices):

    # Make predictions on the unlabeled data
    model.fit(x_labeled, y_labeled)
    predictions = model.predict(x_unlabeled)

    # Generate a report with the predictions and the corresponding indices
    report = pd.DataFrame({
        'index': indices,
        'label': predictions.astype(int)
    })

    report.to_csv(output_path, index=False, header=False)
    return predictions

def main():
    x_labeled, y_labeled, x_unlabeled, indices, scale_pos_weight = loadData()
    model, parameters = computeHyperparameters(x_labeled, y_labeled, scale_pos_weight)
    ber, matrix, error_rates = evaluateModel(model, x_labeled, y_labeled)
    predictions = makePredictions(model, x_labeled, y_labeled, x_unlabeled, indices)

    print("RESULTS: ")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(matrix)
    print(f"\n  Class 0 error rate: {error_rates[0]:.4f}")
    print(f"  Class 1 error rate: {error_rates[1]:.4f}")
    
    # BER
    print(f"\n*** Balanced Error Rate (BER): {ber:.4f} ***")
    print(f"*** Balanced Accuracy: {1-ber:.4f} ***")
    
    # Selected Model and Hyperparameters
    print()
    print("MODEL: XGBoost")
    print("\nOptimal Hyperparameters:")
    for param, value in sorted(parameters.items()):
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    return model, ber

if __name__ == "__main__":
    model, ber = main()