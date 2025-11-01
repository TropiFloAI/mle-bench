import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os

# Load the data
train = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "train.csv"))
test = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "test.csv"))


# CO_DATASCIENTIST_BLOCK_START

# Separate features and target
X = train.drop(columns=["id", "target"])
y = train["target"]
X_test = test.drop(columns=["id"])

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            [col for col in X.columns if col not in categorical_cols],
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Evaluate the model using 5-fold cross-validation
cv_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

# Fit the model on the entire training data
model.fit(X, y)

# CO_DATASCIENTIST_BLOCK_END

print(f"KPI: {cv_auc.mean():.4f}")

# Predict probabilities on the test data
test_probs = model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({"id": test["id"], "target": test_probs})
submission.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)