import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Load image data and labels
image_dir = os.path.join(os.environ.get("DATA_DIR"),"train")
labels_path = os.path.join(os.environ.get("DATA_DIR"),"train_labels.csv")

test_dir = os.path.join(os.environ.get("DATA_DIR"),"test")
submission_file = os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv")

# Load labels
labels_df = pd.read_csv(labels_path)
labels = labels_df['label'].values
image_names = labels_df['id'].values  # Changed 'image_name' to 'id'

# Load images
images = []
valid_labels = []
for image_name, label in zip(image_names, labels):
    image_path = os.path.join(image_dir, f"{image_name}.tif")  # Append '.tif' extension
    if os.path.exists(image_path):
        with Image.open(image_path) as image:
            images.append(image.copy())
        valid_labels.append(label)
    else:
        print(f'Warning: Image file {image_path} does not exist and will be skipped.')

# CO_DATASCIENTIST_BLOCK_START

# 2. Preprocess images
def preprocess_image(image, size=(64, 64)):
    image = image.resize(size)
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array.flatten()  # Flatten the image to a 1D array

processed_images = np.array([preprocess_image(image) for image in images])

# 3. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(processed_images, valid_labels, test_size=0.2, random_state=42)

# 4. Define a basic logistic regression model
model = LogisticRegression(max_iter=1000)

# 5. Train the model and evaluate
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_val, y_pred)

# CO_DATASCIENTIST_BLOCK_END

# Print KPI
print(f"KPI: {auc:.4f}")

# 6. Run the model on the test data
# Load test images
test_images = []
test_image_names = []
for test_image_name in os.listdir(test_dir):
    test_image_path = os.path.join(test_dir, test_image_name)
    if os.path.exists(test_image_path):
        with Image.open(test_image_path) as image:
            test_images.append(preprocess_image(image))
        test_image_names.append(test_image_name.split('.')[0])  # Extract ID without extension
    else:
        print(f'Warning: Test image file {test_image_path} does not exist and will be skipped.')

# Convert test images to numpy array
test_images = np.array(test_images)

# Make predictions on the test set
test_predictions = model.predict_proba(test_images)[:, 1]  # Get probabilities for the positive class
 
# Prepare the submission file
submission_df = pd.DataFrame({"id": test_image_names, "label": test_predictions})
submission_df.to_csv(submission_file, index=False)

print(f"Submission file saved to {submission_file}")


