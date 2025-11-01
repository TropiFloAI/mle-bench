import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_log_error
import os

# Load the data
train = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "train.csv"))
test = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"),"test.csv"))

# CO_DATASCIENTIST_BLOCK_START

# Prepare the features and targets
X = train.drop(columns=["id", "formation_energy_ev_natom", "bandgap_energy_ev"])
y_formation = train["formation_energy_ev_natom"]
y_bandgap = train["bandgap_energy_ev"]

# Initialize models
rf_formation = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_bandgap = RandomForestRegressor(random_state=42, n_jobs=-1)

# Evaluate using cross-validation on full training data
rmsle_formation = np.sqrt(
    -cross_val_score(
        rf_formation,
        X,
        y_formation,
        scoring="neg_mean_squared_log_error",
        cv=5,
    ).mean()
)
rmsle_bandgap = np.sqrt(
    -cross_val_score(
        rf_bandgap, X, y_bandgap, scoring="neg_mean_squared_log_error", cv=5
    ).mean()
)

# Train on full training data for final predictions
rf_formation.fit(X, y_formation)
rf_bandgap.fit(X, y_bandgap)

# Prepare test data
X_test = test.drop(columns=["id"])

# Make predictions
formation_preds = rf_formation.predict(X_test)
bandgap_preds = rf_bandgap.predict(X_test)

# CO_DATASCIENTIST_BLOCK_END

# Take negative of mean RMSLE for KPI that increases
print(f"RMSLE for Formation Energy: {rmsle_formation}")
print(f"RMSLE for Bandgap Energy: {rmsle_bandgap}")
print(f"KPI: {-(rmsle_formation + rmsle_bandgap)}")

# Create submission file
submission = pd.DataFrame(
    {
        "id": test["id"],
        "formation_energy_ev_natom": formation_preds,
        "bandgap_energy_ev": bandgap_preds,
    }
)

# Save submission
submission.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)