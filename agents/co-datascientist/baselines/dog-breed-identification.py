import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.optim as optim
from tqdm import tqdm

# Directories
train_dir = os.path.join(os.environ.get("DATA_DIR"),"train")
test_dir = os.path.join(os.environ.get("DATA_DIR"),"test")
labels_file = os.path.join(os.environ.get("DATA_DIR"),"labels.csv")
sample_submission_file = os.path.join(os.environ.get("DATA_DIR"), "sample_submission.csv")
submission_file = os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv")


# Load labels
labels_df = pd.read_csv(labels_file)
breed_list = labels_df["breed"].unique()
breed_to_idx = {breed: idx for idx, breed in enumerate(breed_list)}

# CO_DATASCIENTIST_BLOCK_START

# Dataset class
class DogBreedDataset(Dataset):
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
        return image


# Transforms - OPTIMIZED: smaller images for speed
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Reduced from 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Prepare data
train_files = [
    os.path.join(train_dir, f"{row['id']}.jpg") for _, row in labels_df.iterrows()
]
train_labels = [breed_to_idx[row["breed"]] for _, row in labels_df.iterrows()]

# # QUICK TEST: Use only first 500 samples (~2 min)
# train_files = train_files[:500]
# train_labels = train_labels[:500]

# Split data
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.2, random_state=42
)

print(f"Training on {len(train_files)} samples, validating on {len(val_files)} samples")
print(f"Number of breeds: {len(breed_list)}")

# Datasets and loaders - OPTIMIZED: larger batches, parallel loading
train_dataset = DogBreedDataset(train_files, train_labels, transform)
val_dataset = DogBreedDataset(val_files, val_labels, transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Model - OPTIMIZED: using ResNet18 instead of ResNet50 for speed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)  # Changed from resnet50
model.fc = nn.Linear(model.fc.in_features, len(breed_list))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_log_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        val_log_loss_score = validate_model(model, val_loader, criterion)
        
        # Early stopping logic
        if val_log_loss_score < best_val_log_loss:
            best_val_log_loss = val_log_loss_score
            patience_counter = 0
            print(f"✓ New best validation log loss!")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return best_val_log_loss


# Validate function
def validate_model(model, val_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.array(all_labels)
    val_log_loss = log_loss(all_labels, all_preds, labels=list(range(len(breed_list))))
    print(f"Validation Log Loss: {val_log_loss:.4f}")
    return val_log_loss


# Train and validate the model
best_val_log_loss = train_model(model, train_loader, val_loader, criterion, optimizer)

# CO_DATASCIENTIST_BLOCK_END

# Print KPI (negative log loss for maximization)
print(f"KPI: {-best_val_log_loss}")

# Prepare test data
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
test_dataset = DogBreedDataset(test_files, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)


# Prediction
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy())
    return np.vstack(predictions)


# Generate predictions
test_predictions = predict(model, test_loader)


# Create submission
submission_df = pd.read_csv(sample_submission_file)
submission_df.iloc[:, 1:] = test_predictions
submission_df.to_csv(submission_file, index=False)

print("Submission file created at:", submission_file)
