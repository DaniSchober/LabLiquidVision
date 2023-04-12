from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from src.data.dataloader import VesselCaptureDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from src.models.model import VesselNet


# Define the training loop
def train(model, criterion, optimizer, train_loader, epoch_str):
    model.train()

    # Wrap train_loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=epoch_str)

    for i, data in enumerate(progress_bar):
        inputs = data["depth_image"]
        targets = torch.stack([data["vol_liquid"], data["vol_vessel"]], dim=1)
        targets = targets.float()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        # Calculate RMSE
        rmse = torch.sqrt(loss).item()
        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item(), "RMSE": rmse})


data_dir = "data/interim/"
batch_size = 2
num_epochs = 10
learning_rate = 0.001

# Load the dataset
dataset = VesselCaptureDataset(data_dir)

# Split the dataset into training and test data
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Set up the data loader and training parameters for the training data
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
train_size = len(train_data)

# Set up the data loader and training parameters for the test data
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
test_size = len(test_data)

model = VesselNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_str = "Epoch " + str(epoch + 1)
    # print(f'Epoch {epoch + 1}/{num_epochs}')
    train(model, criterion, optimizer, train_loader, epoch_str)

# Save the trained model
torch.save(model.state_dict(), "models/vessel_net.pth")
