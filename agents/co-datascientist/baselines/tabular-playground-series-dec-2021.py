import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import os


# Load the data
train = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "train.csv"))
test = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "test.csv"))


# Prepare the features and target
X = train.drop(columns=["Id", "Cover_Type"])
y = train["Cover_Type"]
X_test = test.drop(columns=["Id"])


# CO_DATASCIENTIST_BLOCK_START

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

# Validate
val_accuracy = model.score(X_val, y_val)

# Retrain on full dataset for final predictions
model.fit(X, y)

# CO_DATASCIENTIST_BLOCK_END

# Print KPI (accuracy, higher is better)
print(f"KPI: {val_accuracy:.6f}")

# Make predictions on the test set
predictions = model.predict(X_test)

# Prepare the submission file
submission = pd.DataFrame({"Id": test["Id"], "Cover_Type": predictions})
submission.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
