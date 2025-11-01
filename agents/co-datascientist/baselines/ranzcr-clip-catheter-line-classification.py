import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.optim as optim
from tqdm import tqdm

# Directories
train_dir = os.path.join(os.environ.get("DATA_DIR"), "train")
test_dir = os.path.join(os.environ.get("DATA_DIR"), "test")
train_csv = os.path.join(os.environ.get("DATA_DIR"), "train.csv")
sample_submission_file = os.path.join(os.environ.get("DATA_DIR"), "sample_submission.csv")
submission_file = os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv")

# Load labels
train_df = pd.read_csv(train_csv)
label_cols = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
              'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
              'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
              'Swan Ganz Catheter Present']

# CO_DATASCIENTIST_BLOCK_START

# Dataset class
class RANZCRDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image


# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Prepare data
train_files = [os.path.join(train_dir, f"{row['StudyInstanceUID']}.jpg") for _, row in train_df.iterrows()]
train_labels = train_df[label_cols].values

# # QUICK TEST: Use only first 200 samples for fast testing
# train_files = train_files[:200]
# train_labels = train_labels[:200]

# Split data
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.2, random_state=42
)

# Datasets and loaders
train_dataset = RANZCRDataset(train_files, train_labels, transform)
val_dataset = RANZCRDataset(val_files, val_labels, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
num_classes = len(label_cols)  # Multi-label classification
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    best_val_auc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        val_auc = validate_model(model, val_loader, criterion)
        
        # Track best validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
    
    return best_val_auc


# Validate function
def validate_model(model, val_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate mean ROC AUC across all classes
    auc_scores = []
    for i in range(all_labels.shape[1]):
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        auc_scores.append(auc)
    mean_auc = np.mean(auc_scores)
    print(f"Validation Mean ROC AUC: {mean_auc:.4f}")
    return mean_auc


# Train and validate the model
best_val_auc = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# CO_DATASCIENTIST_BLOCK_END

# Print KPI (mean ROC AUC, higher is better)
print(f"KPI: {best_val_auc}")

# Prepare test data
sample_submission = pd.read_csv(sample_submission_file)
test_files = [os.path.join(test_dir, f"{study_id}.jpg") for study_id in sample_submission['StudyInstanceUID']]
test_dataset = RANZCRDataset(test_files, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Prediction
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predictions.append(probs.cpu().numpy())
    return np.vstack(predictions)


# Generate predictions
test_predictions = predict(model, test_loader)

# Create submission
sample_submission[label_cols] = test_predictions
sample_submission.to_csv(submission_file, index=False)

print("Submission file created at:", submission_file)

