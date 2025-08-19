from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from db import model, jobs_file, JOB_PATH
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from fastapi.responses import RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER
from urllib.parse import urlencode
from fastapi.staticfiles import StaticFiles

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix




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
        'тошкент вилояти': 2800
    }

    cottonseed_oil_prices = {
        'фаргона вилояти': 19750,
        'наманган вилояти': 17100,
        'андижон вилояти': 18500,
        'тошкент шахри': 18250,
        'тошкент вилояти': 18250
    }

    beef_prices = {
        'фаргона вилояти': 77000,
        'наманган вилояти': 76000,
        'андижон вилояти': 82500,
        'тошкент шахри': 85000,
        'тошкент вилояти': 85000
    }

    feature_cols = [
        'семейное положение', 'сумма_выдачи(usd)', 'процентная ставка',
        'срок кредита (месяц)', 'цикл', 'цель кредита', 'возраст', 'пол',
        'область \город', 'регион \город', 'количество членов семьи', 'образование',
        'должность', 'shaped_bread(som)', 'cottonseed oil(som)', 'beef(som)'
    ]

    categorical_cols = [
        'семейное положение', 'цель кредита', 'пол', 'область \город',
        'регион \город', 'образование', 'должность'
    ]

    input_data['shaped_bread(som)'] = input_data['область \\город'].map(shaped_bread_prices)
    input_data['cottonseed oil(som)'] = input_data['область \\город'].map(cottonseed_oil_prices)
    input_data['beef(som)'] = input_data['область \\город'].map(beef_prices)


    df_encoded = input_data.copy()
    label_encoders = {}


    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    X = df_encoded[feature_cols]

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
        "not approved %": round(bad_prob, 2)
    }


@app.get("/")
async def form_page(request: Request, prediction_status: str = None, approved: float = None, not_approved: float = None):
    prediction_data = None
    if prediction_status:
        prediction_data = {
            "prediction_status": prediction_status,
            "approved": approved,
            "not_approved": not_approved,
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
    threshold: int = Form(50), 
):
    threshold = 50
    threshold_normalized = threshold / 100.0
  
    client_data = {
        'семейное положение': marital_status,
        'сумма_выдачи(usd)': loan_amount,
        'процентная ставка': interest_rate,
        'срок кредита (месяц)': loan_term,
        'цикл': cycle,
        'цель кредита': loan_purpose,
        'возраст': age,
        'пол': gender,
        'область \город': region,
        'регион \город': city,
        'количество членов семьи': family_members,
        'образование': education,
        'должность': job_position,
}

    predicted_data = get_prediction(model, client_data, threshold=threshold_normalized)

    query_params = {
        "prediction_status": predicted_data['prediction_status'],
        "approved": predicted_data['approved %'],
        "not_approved": predicted_data['not approved %'],
    }

    url = str(request.url_for("form_page")) + "?" + urlencode(query_params)
    return RedirectResponse(url=url, status_code=HTTP_303_SEE_OTHER)
