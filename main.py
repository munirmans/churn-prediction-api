import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("churn_model.joblib")

app = FastAPI(title="Churn Prediction API")


class Customer(BaseModel):
    senior_citizen: int
    tenure: int
    monthly_charges: float
    total_charges: float
    gender: str
    partner: str
    dependents: str
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str


@app.post("/predict")
def predict(customer: Customer):
    customer_df = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(customer_df)[0]
    # model.classes_ is ['No', 'Yes'] alphabetically, so column 1 = probability of "Yes"
    probability = model.predict_proba(customer_df)[0, 1]

    return {
        "churn_prediction": prediction,
        "churn_probability": float(probability),
    }
