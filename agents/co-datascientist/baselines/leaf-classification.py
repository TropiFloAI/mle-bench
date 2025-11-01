import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import numpy as np

import os

# Load the data
train = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "train.csv"))
test = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"),"test.csv"))
sample_submission = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"),"sample_submission.csv"))

# CO_DATASCIENTIST_BLOCK_START

# Prepare the data
X_train = train.drop(["id", "species"], axis=1)
y_train = train["species"]
X_test = test.drop(["id"], axis=1)

# Encode the labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Evaluate the model using 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_loss_scores = cross_val_score(
    model, X_train, y_train_encoded, cv=cv, scoring="neg_log_loss"
)

# Train the model on the entire training set
model.fit(X_train, y_train_encoded)

# CO_DATASCIENTIST_BLOCK_END

# Log loss scores are negative, aiming to increase to zero
print(f"KPI: {np.mean(log_loss_scores):.4f}")

# Predict probabilities on the test set
y_test_pred_proba = model.predict_proba(X_test)


# Prepare the submission file
submission = pd.DataFrame(y_test_pred_proba, columns=le.classes_)
submission.insert(0, "id", test["id"])
submission.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
