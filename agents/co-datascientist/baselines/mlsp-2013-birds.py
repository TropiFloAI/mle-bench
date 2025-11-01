import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, roc_auc_score

# Load data with proper paths
data_dir = os.environ.get("DATA_DIR")
# Features are comma-separated, first column is rec_id, rest are features
features = pd.read_csv(os.path.join(data_dir, 'supplemental_data/segment_features.txt'), sep=',')
# The header has two columns but actually rec_id is first and rest are features
# Rename first column to rec_id for consistency
if 'rec_id,[histogram of segment features]' in features.columns:
    features = features.rename(columns={'rec_id,[histogram of segment features]': 'rec_id'})
elif features.columns[0] != 'rec_id':
    features = features.rename(columns={features.columns[0]: 'rec_id'})

# Load labels - manually parse because of variable number of comma-separated labels
labels = []
with open(os.path.join(data_dir, 'essential_data/rec_labels_test_hidden.txt'), 'r') as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split(',')
        rec_id = int(parts[0])
        label_list = parts[1:] if len(parts) > 1 else ['?']
        labels.append({'rec_id': rec_id, '[labels]': label_list})

labels = pd.DataFrame(labels)

# CO_DATASCIENTIST_BLOCK_START

# Align features with labels using `rec_id` for mapping
data = pd.merge(features, labels, on='rec_id')

# Handle missing labels denoted by '?' by removing them
data = data[data['[labels]'].apply(lambda x: '?' not in x)]

# Identify feature columns
feature_columns = data.columns.difference(['rec_id', '[labels]'])

# Split data into training and validation BEFORE scaling
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

X_train = train_data[feature_columns]
y_train = train_data['[labels]']
X_val = val_data[feature_columns]
y_val = val_data['[labels]']

# Normalize the feature values - fit only on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Use MultiLabelBinarizer to fit the labels
mlb = MultiLabelBinarizer()
y_train_binarized = mlb.fit_transform(y_train)
y_val_binarized = mlb.transform(y_val)

# Train a logistic regression model for multi-label classification
model = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
model.fit(X_train, y_train_binarized)

# Make predictions on validation set (probabilities for ROC AUC)
val_predictions_proba = model.predict_proba(X_val)
# CO_DATASCIENTIST_BLOCK_END

# Calculate validation ROC AUC (evaluation metric for this task)
val_roc_auc = roc_auc_score(y_val_binarized, val_predictions_proba)

# Print final KPI (higher is better)
print(f"KPI: {val_roc_auc:.6f}")


# Load test data and make predictions
test_features = pd.read_csv(os.path.join(data_dir, 'supplemental_data/segment_features.txt'), sep=',')
# Fix column names
if 'rec_id,[histogram of segment features]' in test_features.columns:
    test_features = test_features.rename(columns={'rec_id,[histogram of segment features]': 'rec_id'})
elif test_features.columns[0] != 'rec_id':
    test_features = test_features.rename(columns={test_features.columns[0]: 'rec_id'})
test_features[feature_columns] = scaler.transform(test_features[feature_columns])

# Predict on test set
X_test = test_features[feature_columns]
test_predictions_binarized = model.predict(X_test)
test_predictions = mlb.inverse_transform(test_predictions_binarized)

# Save predictions
submission_dir = os.environ.get("SUBMISSION_DIR")
submission = pd.DataFrame({
    'rec_id': test_features['rec_id'], 
    'predicted_label': [' '.join(pred) if pred else '' for pred in test_predictions]
})
submission.to_csv(os.path.join(submission_dir, 'submission.csv'), index=False)

