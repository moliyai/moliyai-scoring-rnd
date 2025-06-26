###03.10
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import math


logistic_model = joblib.load("./model/new_logistic_regression_model.pkl")
random_forest_model = joblib.load("./model/new_random_forest_model.pkl")

app = FastAPI()

class ClientData(BaseModel):
    loan_amount: float
    interest_rate: float
    collateral_amount: float
    credit_history_count: int
    overdue_days: int
    age: int
    loan_duration_month: int
    branch: str
    collateral_type: str
    product: str
    education: str
    occupation: str
    marital_status: str
    gender: str
    monthly_income: float  
    risk_adjustment_factor: float 


def calculate_max_loan(monthly_income: float, interest_rate: float, loan_duration_month: int):
    interest_rate = interest_rate / 100
    monthly_interest_rate = interest_rate / 12
    max_monthly_payment = monthly_income * 0.5
    
    if monthly_interest_rate == 0:
        max_loan_amount = max_monthly_payment * loan_duration_month
    else:
        max_loan_amount = max_monthly_payment * (1 - math.pow(1 + monthly_interest_rate, -loan_duration_month)) / monthly_interest_rate
    
    return max_loan_amount


def get_prediction(model, data: ClientData, threshold: float = 0.5):
    input_data = pd.DataFrame([data.dict()])

   
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    good_prob = probability[0][0] * 100 
    bad_prob = probability[0][1] * 100 
    
   
    max_loan = calculate_max_loan(data.monthly_income, data.interest_rate, data.loan_duration_month)
    risk_factor = 1 - data.risk_adjustment_factor 
    adjusted_loan = max_loan * risk_factor

    
    if good_prob / 100 > threshold:
        return {
            "prediction": "good",
            "good %": round(good_prob, 2),
            "bad %": round(bad_prob, 2),
            "maximum_loan_amount": round(max_loan, 2),
            "adjusted_loan_amount": round(adjusted_loan, 2)
        }
    else:
        return {
            "prediction": "bad",
            "good %": round(good_prob, 2),
            "bad %": round(bad_prob, 2),
        }


@app.post("/predict/logistic/regression")
def predict_logistic_regression(data: ClientData, threshold: float = 0.5):
    return get_prediction(logistic_model, data, threshold)


@app.post("/predict/random/forest")
def predict_random_forest(data: ClientData, threshold: float = 0.5):
    return get_prediction(random_forest_model, data, threshold)
