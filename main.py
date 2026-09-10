import dill
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import (FeatureEngineering, TopNGroupTransformer, DataCleaning, FeatureSelector, ScreenHeightClipper)

app = FastAPI()
with open('ga_model.pkl', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int

    utm_source: str | None = None
    utm_medium: str | None = None
    utm_campaign: str | None = None
    utm_adcontent: str | None = None
    utm_keyword: str | None = None

    device_category: str | None = None
    device_os: str | None = None
    device_brand: str | None = None
    device_browser: str | None = None
    device_model: str | None = None

    geo_country: str | None = None
    geo_city: str | None = None

    device_screen_resolution: str | None = None

class Prediction(BaseModel):
    session_id: str
    prediction: int
    probability: float


@app.get("/status")
def status():
    return "I am OK"

@app.get("/version")
def version():
    return model['metadata']

@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame([form.model_dump()])
    try:
        probability = model["model"].predict_proba(df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {e}")

    prediction = int(probability >= 0.5)

    return {
        "session_id": form.session_id,
        "prediction": prediction,
        "probability": round(float(probability), 4)}



