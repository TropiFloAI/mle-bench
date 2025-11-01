import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import os

# Load data
train = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "train.csv"))
test = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"),"test.csv"))

# CO_DATASCIENTIST_BLOCK_START

# Features and target
X = train[["weight"]]
y = train["overweight"]

# Split (optional, for clarity — cross_val_score will handle this too)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

# Define model — keep it super simple
model = GradientBoostingClassifier(random_state=42)

# Evaluate with cross-validation
cv_scores = cross_val_score(model, X, y, cv=3)

# Fit the model on all training data
model.fit(X, y)

# Prepare test features
X_test = test[["weight"]]


# CO_DATASCIENTIST_BLOCK_END

print(f"KPI: {cv_scores.mean()}")
predictions = model.predict(X_test)


# Prepare submission
submission = pd.DataFrame(
    {"id": test["id"], "overweight": predictions}
)
submission.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
