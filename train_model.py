import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Create model folder if it doesn't exist
os.makedirs("Model", exist_ok=True)

# Load dataset
data = pd.read_csv("C:/Users/smiti/OneDrive/Desktop/AQI prediction/dataset/city_day.csv")

# Select important columns
data = data[['PM2.5','PM10','NO','NO2','NH3','CO','SO2','O3','AQI']]

# Remove missing values
data = data.dropna()

# Features and target
X = data.drop("AQI", axis=1)
y = data["AQI"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Save trained model using joblib
joblib.dump(model, "Model/aqi_model.pkl", compress=3)

print("Model trained and saved successfully!")
