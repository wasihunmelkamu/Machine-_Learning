# src/predict.py
import joblib
import numpy as np
from geopy.distance import distance

def predict_house_price(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
    # Load assets
    model = joblib.load("models/house_price_model.pkl")
    feature_order = joblib.load("models/feature_order.pkl")
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    
    # Same feature engineering as in train.py
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
        data[f'Distance_to_{city}'] = distance((Latitude, Longitude), coords).km
    
    data['Income_x_HouseAge'] = data['MedInc'] * data['HouseAge']
    data['Income_per_Room'] = data['MedInc'] / data['AveRooms']
    
    # Dummy clusters (assign to cluster 0)
    for i in range(15):
        data[f'Cluster_{i}'] = 1.0 if i == 0 else 0.0
    
    # Predict
    input_array = np.array([data[col] for col in feature_order]).reshape(1, -1)
    pred = model.predict(input_array)[0]
    return round(float(pred * 100_000), 2)

if __name__ == "__main__":
    # Example prediction
    price = predict_house_price(
        MedInc=5.0,
        HouseAge=20,
        AveRooms=6.0,
        AveBedrms=1.2,
        Population=1000,
        AveOccup=3.0,
        Latitude=37.8,
        Longitude=-122.2
    )
    print(f"💰 Predicted House Price: ${price:,.2f}")