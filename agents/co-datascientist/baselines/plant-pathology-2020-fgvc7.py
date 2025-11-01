import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "train.csv"))
test_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "test.csv"))
img_dir = os.path.join(os.environ.get("DATA_DIR"), "images")

# CO_DATASCIENTIST_BLOCK_START

# Define dataset class
class PlantDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)
        labels = self.dataframe.iloc[idx, 1:].values.astype("float")
        if self.transform:
            image = self.transform(image)
        return image, labels


# Transformations - REDUCED image size for speed
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Reduced from 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Split data for training and validation
train_sub, val_sub = train_test_split(train_df, test_size=0.2, random_state=42)

print(f"Training on {len(train_sub)} samples, validating on {len(val_sub)} samples")

# Create dataloaders - OPTIMIZED for speed
train_loader = DataLoader(
    PlantDataset(train_sub, img_dir, transform=transform),
    batch_size=64,  # Increased from 32
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    PlantDataset(val_sub, img_dir, transform=transform),
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Model, loss, and optimizer
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with early stopping
epochs = 20
patience = 5
best_auc = 0.0
patience_counter = 0

for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")
    
    # Validation
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    val_auc = roc_auc_score(all_labels, all_preds, average="macro")
    print(f"Validation AUC: {val_auc:.4f}")
    
    # Early stopping logic
    if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0
        print(f"✓ New best AUC!")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")
    
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# CO_DATASCIENTIST_BLOCK_END

# Print KPI (AUC, higher is better)
print(f"KPI: {best_auc:.6f}")


# Predict on test set
test_dataset = PlantDataset(test_df, img_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)
        predictions.append(preds.cpu().numpy())
predictions = np.vstack(predictions)

# Create submission file
submission_df = pd.DataFrame(
    predictions, columns=["healthy", "multiple_diseases", "rust", "scab"]
)
submission_df.insert(0, "image_id", test_df["image_id"])
submission_df.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
