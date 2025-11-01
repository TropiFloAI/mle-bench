import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os

# Load the dataset
train_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "ru_train.csv.zip"))
test_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "ru_test_2.csv.zip"))

# CO_DATASCIENTIST_BLOCK_START

# Split the training data for validation
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

# Create a dictionary of rules based on the training data
rules = defaultdict(str)
for _, row in train_data.iterrows():
    before = row["before"]
    after = row["after"]
    if before not in rules:
        rules[before] = after


# Function to apply rules to a dataset
def apply_rules(df, rules):
    predictions = []
    for _, row in df.iterrows():
        before = row["before"]
        if before in rules:
            predictions.append(rules[before])
        else:
            predictions.append(before)  # Default to 'before' if no rule is found
    return predictions


# Evaluate on validation set
val_predictions = apply_rules(val_data, rules)
val_accuracy = sum(val_predictions == val_data["after"]) / len(val_data)

# CO_DATASCIENTIST_BLOCK_END

print(f"KPI: {val_accuracy:.4f}")

# Predict on the test set
test_predictions = apply_rules(test_df, rules)

# Prepare submission
submission_df = pd.DataFrame(
    {
        "id": test_df["sentence_id"].astype(str)
        + "_"
        + test_df["token_id"].astype(str),
        "after": test_predictions,
    }
)


submission_df.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
