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
X = train.drop(columns=["Transported", "Name"])
y = train["Transported"]
X_test = test.drop(columns=["Name"])

# Preprocessing
numeric_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
numeric_transformer = SimpleImputer(strategy="median")

categorical_features = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42)),
    ]
)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

# Train on full data
model.fit(X, y)

# CO_DATASCIENTIST_BLOCK_END

print(f"KPI: {cv_scores.mean()}")
predictions = model.predict(X_test)


# Prepare submission
submission = pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Transported": predictions}
)
submission.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
