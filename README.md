# Loan Default Prediction Using Machine Learning

## Project Overview
This project builds a machine learning model to predict loan default using borrower financial and credit-related features. The goal is to help financial institutions identify high-risk borrowers and improve credit risk management.

## Business Objective
To develop a predictive model that identifies borrowers likely to default, enabling data-driven lending decisions and minimizing financial risk.

## Dataset
- Source: LendingClub Dataset (Kaggle)
- Sample Size Used: 100,000 records
- Target Variable:
  - 0 → Fully Paid
  - 1 → Charged Off (Default)

The dataset represents an imbalanced classification problem (~20% defaulters).

The dataset used in this project is the LendingClub Loan Dataset available on Kaggle.

Due to file size limitations (383 MB), the dataset is not included in this repository.

You can download the dataset from:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

After downloading, place the dataset file in the project directory before running the notebook.

## Data Preprocessing
- Target cleaning
- Feature selection
- Missing value handling
- Feature engineering (FICO score creation)
- Ordinal encoding (grade)
- One-hot encoding (nominal variables)
- Stratified train-test split
- Feature scaling (for Logistic Regression)

## Models Implemented

### Logistic Regression
- ROC-AUC ≈ 0.71
- Recall (Default) improved to 73% after threshold tuning
- Selected as final model

### Random Forest
- ROC-AUC ≈ 0.70
- Higher accuracy but poor recall for defaulters

## Model Evaluation
- Classification Report
- ROC Curve
- Threshold Tuning
- 5-Fold Cross-Validation (Mean AUC ≈ 0.708, Low variance)

## Key Features Influencing Default
- Loan Term
- Credit Grade
- FICO Score
- Interest Rate
- Debt-to-Income Ratio
- Annual Income

## Conclusion
Logistic Regression with threshold tuning was selected as the final model due to its balanced performance and interpretability. The model demonstrates stable performance across cross-validation folds and effectively identifies high-risk borrowers.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
