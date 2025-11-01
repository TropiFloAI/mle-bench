import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, sqrt, atan2
import os


# Load data
train_df = pd.read_csv(
    os.path.join(os.environ.get("DATA_DIR"), "labels.csv"), nrows=1000000
)  # Use a subset for faster processing
test_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "test.csv"))

# CO_DATASCIENTIST_BLOCK_START

def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371  # Radius of earth in kilometers
    return c * r


# Preprocess data
train_df.dropna(inplace=True)
train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] < 500)]
train_df = train_df[
    (train_df["passenger_count"] > 0) & (train_df["passenger_count"] < 7)
]

train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["day"] = train_df["pickup_datetime"].dt.dayofweek
train_df["month"] = train_df["pickup_datetime"].dt.month

test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["day"] = test_df["pickup_datetime"].dt.dayofweek
test_df["month"] = test_df["pickup_datetime"].dt.month

train_df["distance"] = train_df.apply(
    lambda row: haversine(
        row["pickup_longitude"],
        row["pickup_latitude"],
        row["dropoff_longitude"],
        row["dropoff_latitude"],
    ),
    axis=1,
)
test_df["distance"] = test_df.apply(
    lambda row: haversine(
        row["pickup_longitude"],
        row["pickup_latitude"],
        row["dropoff_longitude"],
        row["dropoff_latitude"],
    ),
    axis=1,
)

features = ["hour", "day", "month", "passenger_count", "distance"]
X = train_df[features]
y = train_df["fare_amount"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# CO_DATASCIENTIST_BLOCK_END

# Validation
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse}")
print(f"KPI: {-rmse}")

# Test predictions
test_preds = model.predict(test_df[features])

# Save submission
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_preds})
submission.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
