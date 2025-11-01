import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import os

# Load data
with open(os.path.join(os.environ.get("DATA_DIR"),"train.json")) as f:
    train_data = json.load(f)

with open(os.path.join(os.environ.get("DATA_DIR"),"test.json")) as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# CO_DATASCIENTIST_BLOCK_START

# Select features and target
features = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "giver_username_if_known",
]

X = train_df[features]
y = train_df["requester_received_pizza"]

# Handle missing values
X.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# Encode categorical variables using pandas factorize
X["giver_username_if_known"], _ = pd.factorize(train_df["giver_username_if_known"])
test_df["giver_username_if_known"] = pd.factorize(test_df["giver_username_if_known"])[0]

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model
val_predictions = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_predictions)

# Train on full training data
model.fit(X, y)

# CO_DATASCIENTIST_BLOCK_END

print(f"KPI: {val_auc}")

# Predict on test data
test_X = test_df[features]
test_predictions = model.predict_proba(test_X)[:, 1]


# Prepare submission
submission = pd.DataFrame(
    {"request_id": test_df["request_id"], "requester_received_pizza": test_predictions}
)

submission.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
