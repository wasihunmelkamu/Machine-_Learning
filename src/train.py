



# src/train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
from geopy.distance import distance

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("📥 Loading dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv("data/california_housing.csv", index=False)
print(f"✅ Dataset saved: {df.shape}")

# === EDA & Visualization (optional: comment out if headless) ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['MedHouseVal'], kde=True)
plt.title('House Price Distribution')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='MedInc', y='MedHouseVal', alpha=0.5)
plt.title('Income vs Price')
plt.tight_layout()
plt.savefig("data/eda.png")
print("📊 EDA plot saved: data/eda.png")

# === Advanced Feature Engineering ===
print("🔧 Engineering features...")

df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']

cities = {
    'Los_Angeles': (34.0522, -118.2437),
    'San_Francisco': (37.7749, -122.4194),
    'San_Diego': (32.7157, -117.1611),
    'Sacramento': (38.5816, -121.4944)
}

for city, coords in cities.items():
    df[f'Distance_to_{city}'] = df.apply(
        lambda row: distance((row['Latitude'], row['Longitude']), coords).km,
        axis=1
    )

df['Income_x_HouseAge'] = df['MedInc'] * df['HouseAge']
df['Income_per_Room'] = df['MedInc'] / df['AveRooms']

# Clustering
print("📍 Adding location clusters...")
coords = df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=15, random_state=42)
df['LocationCluster'] = kmeans.fit_predict(coords)
df = pd.get_dummies(df, columns=['LocationCluster'], prefix='Cluster')

# Prepare data
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Save feature order for API consistency
joblib.dump(X.columns.tolist(), "models/feature_order.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale (for compatibility, though XGBoost doesn't require it)
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, "models/scaler.pkl")

# Train
print("🧠 Training XGBoost model...")
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n✅ Results:\nRMSE: {rmse:.3f}\nR²: {r2:.3f}")

# Save
joblib.dump(model, "models/house_price_model.pkl")
print("💾 Model saved to models/")