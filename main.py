import joblib
import xgboost
import pandas as pd
import json
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

stops_mapping = {"zero": 0, "one": 1, "two_or_more": 2}
class_mapping = {"Business": 1, "Economy": 0}

class FlightInput(BaseModel):
    airline: str
    source_city: str
    destination_city: str
    stops: str
    class_type: str
    duration: float
    days_left: int
    departure_time: str
    arrival_time: str
    model_name: str

with open("Model_metrics/columns.json", "r") as f:
    training_columns = json.load(f)

@app.get('/')
def health():
    return{"Message": "Hello World"}

@app.get('/models')
def list_models():
    return{"avialable_models": ["Linear Regression", "Random Forest", "Decision Tree", "Xgboost"]}

@app.post('/predict')
def predict(data: FlightInput):
    data_cols= {
        'airline': data.airline,
        'source_city': data.source_city,
        'destination_city': data.destination_city,
        'stops': stops_mapping[data.stops],
        'class': class_mapping[data.class_type],
        'duration': data.duration,
        'days_left': data.days_left,
        'departure_time': data.departure_time,
       'arrival_time': data.arrival_time
    }

    df = pd.DataFrame([data_cols])

    df = df.join(pd.get_dummies(df.airline, prefix='airline', dtype=int)).drop('airline', axis=1)
    df = df.join(pd.get_dummies(df.source_city, prefix='source', dtype=int)).drop('source_city', axis=1)
    df = df.join(pd.get_dummies(df.destination_city, prefix='destination', dtype=int)).drop('destination_city', axis=1)
    df = df.join(pd.get_dummies(df.arrival_time, prefix='arrival', dtype=int)).drop('arrival_time', axis =1)
    df = df.join(pd.get_dummies(df.departure_time, prefix='departure', dtype=int)).drop('departure_time', axis=1)
    
    df = df.reindex(columns=training_columns, fill_value=0)

    model = joblib.load(f"Model_metrics/{data.model_name}.pkl")

    prediction = model.predict(df)

    with open(f"Model_metrics/{data.model_name}.json") as f: metrics = json.load(f)

    return {"predicted_price": round(float(prediction[0]), 2), "model_used": data.model_name, "model_r2": metrics["R2 score"]}
