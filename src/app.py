# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from geopy.distance import distance

app = FastAPI(title="House Price Prediction API")

# Load once at startup
model = joblib.load("models/house_price_model.pkl")
feature_order = joblib.load("models/feature_order.pkl")

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        data = features.dict()
        data['RoomsPerHousehold'] = data['AveRooms'] / data['AveOccup']
        data['BedroomsPerRoom'] = data['AveBedrms'] / data['AveRooms']
        data['PopulationPerHousehold'] = data['Population'] / data['AveOccup']
        
        cities = {
            'Los_Angeles': (34.0522, -118.2437),
            'San_Francisco': (37.7749, -122.4194),
            'San_Diego': (32.7157, -117.1611),
            'Sacramento': (38.5816, -121.4944)
        }
        for city, coords in cities.items():
            data[f'Distance_to_{city}'] = distance(
                (data['Latitude'], data['Longitude']), coords
            ).km
        
        data['Income_x_HouseAge'] = data['MedInc'] * data['HouseAge']
        data['Income_per_Room'] = data['MedInc'] / data['AveRooms']
        
        for i in range(15):
            data[f'Cluster_{i}'] = 1.0 if i == 0 else 0.0
        
        input_array = np.array([data[col] for col in feature_order]).reshape(1, -1)
        pred = model.predict(input_array)[0]
        return {"predicted_price_usd": round(float(pred * 100_000), 2)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "OK"}