import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
import numpy as np

import os 

# Load data
train = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "train.csv"))
test = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"),"test.csv"))

# CO_DATASCIENTIST_BLOCK_START

# Prepare features and labels
X_train = train["text"]
y_train = train["author"]
X_test = test["text"]

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Logistic Regression model
model = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")

# Create a pipeline
pipeline = make_pipeline(vectorizer, model)

# Evaluate with 5-fold cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="neg_log_loss")

# Train on full training data
pipeline.fit(X_train, y_train)

# Predict probabilities for test set
probs = pipeline.predict_proba(X_test)

# CO_DATASCIENTIST_BLOCK_END

print(f"KPI: {np.mean(cv_scores):.4f}")


# Prepare submission
submission = pd.DataFrame(probs, columns=pipeline.classes_)
submission.insert(0, "id", test["id"])
submission.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)

