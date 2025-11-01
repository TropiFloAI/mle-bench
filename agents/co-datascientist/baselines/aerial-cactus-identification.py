import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"),"train.csv"))
test_df = pd.read_csv(os.path.join(os.environ.get("DATA_DIR"), "sample_submission.csv"))

# CO_DATASCIENTIST_BLOCK_START

# Define Dataset class
class CactusDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


# Data transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_index, val_index in kf.split(train_df, train_df["has_cactus"]):
    train_data, val_data = train_df.iloc[train_index], train_df.iloc[val_index]

    train_dataset = CactusDataset(
        dataframe=train_data, root_dir=os.path.join(os.environ.get("DATA_DIR"),"train"), transform=transform
    )
    val_dataset = CactusDataset(
        dataframe=val_data, root_dir=os.path.join(os.environ.get("DATA_DIR"),"train"), transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model, criterion and optimizer for each fold
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(5):  # Simple training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validate the model
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_preds = np.array(val_preds).flatten()
    val_labels = np.array(val_labels).flatten()
    auc_score = roc_auc_score(val_labels, val_preds)
    auc_scores.append(auc_score)

# CO_DATASCIENTIST_BLOCK_END

print(f"KPI: {np.mean(auc_scores)}")

# Prepare test data loader
test_dataset = CactusDataset(
    dataframe=test_df, root_dir=os.path.join(os.environ.get("DATA_DIR"),"test"), transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Make predictions on test set
model.eval()
test_preds = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        test_preds.extend(outputs.cpu().numpy())


# Prepare submission file
test_df["has_cactus"] = np.array(test_preds).flatten()
test_df.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
