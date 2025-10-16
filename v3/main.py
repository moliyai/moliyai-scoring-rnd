from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER
from urllib.parse import urlencode
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

import os
import pandas as pd
import numpy as np

from db import model

if not hasattr(np, "bool"):
    np.bool = np.bool_

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
 
USER_INPUT_CSV = "predicted_client_status.csv"

def get_prediction(model, data, threshold = 0):
    input_data = pd.DataFrame([data])

    shaped_bread_prices = {
        'фаргона вилояти': 2650,
        'наманган вилояти': 2700,
        'андижон вилояти': 2700,
        'тошкент шахри': 2800,
        'тошкент вилояти': 2800,
        'самарканд вилояти': 2750,
        'сурхондарё вилояти': 2800,
        'навоий вилояти': 2500,
        'сирдарё вилояти': 2650,
        'кашкадарё вилояти': 2500
    }

    cottonseed_oil_prices = {
        'фаргона вилояти': 19750,
        'наманган вилояти': 17100,
        'андижон вилояти': 18500,
        'тошкент шахри': 18250,
        'тошкент вилояти': 18250,

        'сурхондарё вилояти': 22000,
        'навоий вилояти': 22500,
        'сирдарё вилояти': 21500,
        'кашкадарё вилояти': 25500
    }

    beef_prices = {
        'фаргона вилояти': 77000,
        'наманган вилояти': 76000,
        'андижон вилояти': 82500,
        'тошкент шахри': 85000,
        'тошкент вилояти': 85000,

        'сурхондарё вилояти': 82000,
        'навоий вилояти': 83000,
        'сирдарё вилояти': 80000,
        'кашкадарё вилояти': 75000
    }

    feature_cols = [
        'marital_status', 'loan_amount', 'interest_rate', 'loan_term', 'cycle',
       'loan_purpose', 'age', 'gender', 'region', 'city', 'family_members',
       'education', 'job_position', 'shaped_bread(som)', 'cottonseed oil(som)',
       'beef(som)'
    ]

    categorical_cols = [
        'marital_status', 'loan_purpose', 'gender', 'region', 'city', 'education', 'job_position'
    ]

    input_data['shaped_bread(som)'] = input_data['region'].map(shaped_bread_prices)
    input_data['cottonseed oil(som)'] = input_data['region'].map(cottonseed_oil_prices)
    input_data['beef(som)'] = input_data['region'].map(beef_prices)


    df_encoded = input_data.copy()    
    X = df_encoded[feature_cols]
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    #вероятность класса "1" (плохой клиент)
    y_proba_bad = model.predict_proba(X)[:, 1][0]
    y_proba_good = 1 - y_proba_bad

    bad_prob = y_proba_bad * 100
    good_prob = y_proba_good * 100

    status = "Approved" if y_proba_bad < threshold else "Not Approved"
    
    if status == "Approved":
        client_status = 0
    else:
        client_status = 1
    
    input_data['predicted_client_status'] = client_status
    

    if not os.path.exists(USER_INPUT_CSV):
        input_data.to_csv(USER_INPUT_CSV, index=False)
    else:
        input_data.to_csv(USER_INPUT_CSV, mode='a', header=False, index=False)

    return {
        "prediction_status": status,
        "approved %": round(good_prob, 2),
        "not approved %": round(bad_prob, 2),
        # "shap_plot": f"/static/{shap_bar_filename}" 
    }


@app.get("/")
async def form_page(request: Request, prediction_status: str = None, approved: float = None, not_approved: float = None):
    prediction_data = None
    if prediction_status:
        prediction_data = {
            "prediction_status": prediction_status,
            "approved": approved,
            "not_approved": not_approved,
            # "shap_plot": shap_plot
            
        }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction_data
    })


@app.post("/v3/predict/random/forest")
def predict_random_forest(
    request: Request,
    marital_status: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    region: str = Form(...),
    city: str = Form(...),
    family_members: int = Form(...),
    education: str = Form(...),
    job_position: str = Form(...),
    loan_amount: int = Form(...),
    interest_rate: int = Form(...),
    loan_term: int = Form(...), #срок кредита
    cycle: int = Form(...),
    loan_purpose: str = Form(...),
    threshold: int = Form(60), 
):
    threshold = 50
    threshold_normalized = threshold / 100.0
  

    client_data = {
        'marital_status': marital_status,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'loan_term': loan_term,
        'cycle': cycle,
        'loan_purpose': loan_purpose,
        'age': age,
        'gender': gender,
        'region': region,
        'city': city,
        'family_members': family_members,
        'education': education,
        'job_position': job_position,
    }

    predicted_data = get_prediction(model, client_data, threshold=threshold_normalized)

    query_params = {
        "prediction_status": predicted_data['prediction_status'],
        "approved": predicted_data['approved %'],
        "not_approved": predicted_data['not approved %'],
        # "shap_plot": predicted_data['shap_plot']
    }

    url = str(request.url_for("form_page")) + "?" + urlencode(query_params)
    return RedirectResponse(url=url, status_code=HTTP_303_SEE_OTHER)
