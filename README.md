# Student Admission Prediction using XGBoost

## Overview
An end-to-end ML application that predicts **Chance of Admit** from graduate admission features and deploys the trained XGBoost regression model through Streamlit.

## Dataset
- Supplied dataset: `data/graduate_admission.csv`
- Records used: 500
- Inputs: GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research
- Target: `Chance of Admit`
- Public reference: https://www.kaggle.com/datasets/mohansacharya/graduate-admissions

## Why Regression?
`Chance of Admit` is a continuous value from 0 to 1, so the correct formulation is regression. The Streamlit app converts the prediction to a percentage.

## Workflow
1. Load and inspect data
2. Remove index/identifier columns
3. Clean column names
4. Validate numeric fields and handle missing rows
5. Perform EDA
6. 80/20 train-test split
7. Tune XGBoost using 5-fold CV
8. Evaluate with MAE, RMSE and R²
9. Save model with Joblib
10. Deploy with Streamlit

## Test Performance
- MAE: 0.0427
- RMSE: 0.0628
- R²: 0.8071
- Best parameters: `{'colsample_bytree': 0.8, 'learning_rate': 0.02, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}`

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Project Structure
```text
student_admission_prediction_xgboost/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/graduate_admission.csv
├── models/admission_xgboost_model.joblib
├── models/model_metadata.json
├── assets/*.png
└── reports/
```

## Limitations
The model is an educational estimate based on historical data. It is not an official admission decision, and performance may change on new institutions or populations.

## Academic Integrity
Review and understand the code, test it yourself, and personalize the presentation/report before submission.
