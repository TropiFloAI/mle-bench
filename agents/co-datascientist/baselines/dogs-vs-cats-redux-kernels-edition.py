import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
train_dir = os.path.join(os.environ.get("DATA_DIR"), "train")
test_dir = os.path.join(os.environ.get("DATA_DIR"), "test")
submission_path = os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv")

# CO_DATASCIENTIST_BLOCK_START

# Image transformations
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


# Prepare data
train_files = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir)]
train_labels = [1 if "dog" in fname else 0 for fname in os.listdir(train_dir)]


train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.2, random_state=42
)

print(f"Training on {len(train_files)} samples, validating on {len(val_files)} samples")

# Dataloaders - INCREASED batch size for speed
train_dataset = CustomDataset(train_files, train_labels, transform)
val_dataset = CustomDataset(val_files, val_labels, transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Define CNN model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
from tqdm import tqdm

epochs = 20  # More epochs with early stopping
best_val_loss = float('inf')
patience = 5  # Early stopping patience
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")
    
    # Validate the model
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.numpy())

    val_loss = log_loss(val_targets, val_preds)
    print(f"Validation Log Loss: {val_loss:.4f}")
    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print(f"✓ New best validation loss!")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")
        
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# CO_DATASCIENTIST_BLOCK_END

# Print KPI (negative log loss for maximization)
print(f"KPI: {-best_val_loss}")

# Prepare test data
test_files = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir)]
test_dataset = CustomDataset(test_files, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Generate predictions
model.eval()
test_preds = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs).cpu().numpy()
        test_preds.extend(preds)

# Create submission file
submission_df = pd.DataFrame(
    {"id": range(1, len(test_preds) + 1), "label": np.array(test_preds).flatten()}
)
submission_df.to_csv(submission_path, index=False)
