import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import numpy as np
import os

# Load data
train_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "train.csv"))
test_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"),"test.csv"))

# CO_DATASCIENTIST_BLOCK_START

# Define features and target
X_train = train_df["comment_text"]
y_train = train_df[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
]

# Text preprocessing and model pipeline
pipeline = make_pipeline(
    TfidfVectorizer(max_features=10000, stop_words="english"),
    OneVsRestClassifier(LogisticRegression(solver="saga", max_iter=1000)),
)

# Cross-validation to get KPI
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
mean_cv_score = np.mean(cv_scores)

# Fit the model on the entire training data for final predictions
pipeline.fit(X_train, y_train)

# Predict on test data
X_test = test_df["comment_text"]
predictions = pipeline.predict_proba(X_test)

# CO_DATASCIENTIST_BLOCK_END

# Print KPI
print(f"KPI: {mean_cv_score}")

# Prepare submission
submission_df = pd.DataFrame(
    predictions,
    columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
)
submission_df.insert(0, "id", test_df["id"])
submission_df.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
