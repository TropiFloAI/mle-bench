import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# CO_DATASCIENTIST_BLOCK_START

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Custom dataset class
class DocumentDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.noisy_images = os.listdir(noisy_dir)

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.noisy_images[idx])
        noisy_image = Image.open(noisy_path).convert("L")
        clean_image = Image.open(clean_path).convert("L")
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        return noisy_image, clean_image


# Data preparation
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

train_noisy_dir = os.path.join(os.environ.get("DATA_DIR"), "train/")
train_clean_dir = os.path.join(os.environ.get("DATA_DIR"), "train_cleaned/")
dataset = DocumentDataset(train_noisy_dir, train_clean_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Test data preparation
test_dir = os.path.join(os.environ.get("DATA_DIR"), "test/")
test_images = os.listdir(test_dir)
test_transform = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)


# Model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
best_val_rmse = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * noisy.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation loop - compute RMSE
    model.eval()
    val_mse = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            val_mse += loss.item() * noisy.size(0)
    val_mse /= len(val_loader.dataset)
    val_rmse = np.sqrt(val_mse)
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}")
    
    # Submissions are evaluated on the root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

# CO_DATASCIENTIST_BLOCK_END

# Print KPI (negative RMSE for maximization)
print(f"KPI: {-best_val_rmse}")

# Make predictions
model.eval()
predictions = []
with torch.no_grad():
    for image_name in test_images:
        image_path = os.path.join(test_dir, image_name)
        image = Image.open(image_path).convert("L")
        image = test_transform(image).unsqueeze(0).to(device)
        output = model(image).squeeze().cpu().numpy()
        output = (output * 255).astype(np.uint8)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                id_str = f"{image_name.split('.')[0]}_{i+1}_{j+1}"
                value = output[i, j] / 255.0
                predictions.append([id_str, value])



# Save predictions to submission file
submission_df = pd.DataFrame(predictions, columns=["id", "value"])
submission_df.to_csv(os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv"), index=False)
