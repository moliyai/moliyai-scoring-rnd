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




app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
 

USER_INPUT_CSV = "user_inputs.csv"


def get_maximum_available_amount(salary, total_client_family_salary, client_monthly_expenditure):
    max_available_amount = ((salary + total_client_family_salary - client_monthly_expenditure) / 2) * 12
    return max_available_amount


def get_prediction(model, data, threshold = 0):
    input_data = pd.DataFrame([data])

    salary = input_data["salary"].iloc[0]
    total_client_family_salary = input_data["total_client_family_salary"].iloc[0]
    client_monthly_expenditure = input_data["client_monthly_expenditure"].iloc[0]
   
    probability = model.predict_proba(input_data)
    prediction = model.predict(input_data)

    status = prediction[0]
    input_data['status'] = status


    cancelled_prob = probability[0][0]
    approved_prob = probability[0][1]
    

    if approved_prob >= threshold:
        maximum_available_amount = int(round(approved_prob, 2) * get_maximum_available_amount(salary, total_client_family_salary, client_monthly_expenditure))
        if maximum_available_amount <= 0:
            status = 0
            status_text = "not approved"
            maximum_available_amount = 0
        else:
            status = 1  
            status_text = "approved"
            maximum_available_amount = maximum_available_amount
    else:
        status = 0 
        status_text = "not approved"
        maximum_available_amount = 0


    input_data["max_available_amount_year"] = maximum_available_amount
    if not os.path.exists(USER_INPUT_CSV):
        input_data.to_csv(USER_INPUT_CSV, index=False)
    else:
        input_data.to_csv(USER_INPUT_CSV, mode='a', header=False, index=False)


    return {
        "prediction_status": status_text,
        "not approved %": round(cancelled_prob*100, 2),
        "approved %": round(approved_prob*100, 2),
        "threshold": threshold,
        "maximum_available_amount": maximum_available_amount
    }

@app.get("/")
async def form_page(request: Request, prediction_status: str = None, approved: float = None, not_approved: float = None, maximum_available_amount: float = None):
    prediction_data = None
    if prediction_status:
        prediction_data = {
            "prediction_status": prediction_status,
            "approved": approved,
            "not_approved": not_approved,
            "maximum_available_amount": maximum_available_amount,
        }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction_data
    })



@app.post("/v3/predict/random/forest")
def predict_random_forest(
    request: Request,
    sex: int = Form(...),
    viloyat_id: int = Form(...),
    tuman_id: int = Form(...),
    mfy_id: int = Form(...),
    client_type: int = Form(...),
    job_name: str = Form(...),
    salary: int = Form(...),
    total_client_family_salary: int = Form(...),
    occupation_status: int = Form(...),
    family_status: int = Form(...),
    family_members_count: int = Form(...),
    home_type: int = Form(...),
    home_owner: int = Form(...),
    client_monthly_expenditure: int = Form(...),
    product_price: int = Form(...),
    final_produt_price: int = Form(...),
    threshold: int = Form(80), 
):
    
    threshold_normalized = threshold / 100.0
  
    client_data = {
        "sex": sex,
        "viloyat_id": viloyat_id,
        "tuman_id": tuman_id,
        "mfy_id": mfy_id,
        "client_type": client_type,
        "job_name": job_name,
        "salary": salary,
        "total_client_family_salary": total_client_family_salary,
        "occupation_status": occupation_status,
        "family_status": family_status,
        "family_members_count": family_members_count,
        "home_type": home_type,
        "home_owner": home_owner,
        "client_monthly_expenditure": client_monthly_expenditure,
        "product_price": product_price,
        "final_produt_price": final_produt_price,
    }

    predicted_data = get_prediction(model, client_data, threshold=threshold_normalized)
    print("PREDICAED DATA:", predicted_data['maximum_available_amount'])

    query_params = {
        "prediction_status": predicted_data['prediction_status'],
        "approved": predicted_data['approved %'],
        "not_approved": predicted_data['not approved %'],
        "maximum_available_amount": predicted_data['maximum_available_amount']
    }

    url = str(request.url_for("form_page")) + "?" + urlencode(query_params)
    
    return RedirectResponse(url=url, status_code=HTTP_303_SEE_OTHER)
