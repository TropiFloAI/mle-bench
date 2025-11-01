import pandas as pd
from sklearn.model_selection import train_test_split

import os 


# Load data
train_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "en_train.csv.zip"))
test_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"),"en_test_2.csv.zip"))


# CO_DATASCIENTIST_BLOCK_START

# Split train data into train and validation sets
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)


# Define a simple rule-based function for each class
def normalize_token(row):
    if row["class"] == "PLAIN":
        return row["before"]
    elif row["class"] == "PUNCT":
        return row["before"]
    elif row["class"] == "DATE":
        # Example rule: convert dates like '6/7/2020' to 'June 7, 2020'
        # This is a simplified example; real rules would be more complex
        try:
            return pd.to_datetime(row["before"]).strftime("%B %d, %Y")
        except:
            return row["before"]
    elif row["class"] == "LETTERS":
        # Example rule: convert letters like 'abc' to 'A B C'
        return " ".join(list(row["before"].upper()))
    else:
        return row["before"]


# Apply normalization to validation data
val_data["predicted_after"] = val_data.apply(normalize_token, axis=1)

# Calculate accuracy on validation set
accuracy = (val_data["predicted_after"] == val_data["after"]).mean()

# CO_DATASCIENTIST_BLOCK_END

print(f"KPI: {accuracy:.4f}")

# Apply normalization to test data (assume 'class' can be inferred or default to 'PLAIN')
test_df["after"] = test_df["before"]  # Default to identity function for simplicity


# Save the predictions to submission file
submission = test_df[["sentence_id", "token_id", "after"]].copy()
submission["id"] = (
    submission["sentence_id"].astype(str) + "_" + submission["token_id"].astype(str)
)
submission[["id", "after"]].to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
