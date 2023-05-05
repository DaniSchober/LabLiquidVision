from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from src.data.dataloader import VesselCaptureDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from src.models.model_new import VolumeNet


# Define the training loop
def train(model, criterion, optimizer, train_loader, epoch_str):
    model.train()

    # Wrap train_loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=epoch_str)

    rmse_epoch = 0

    for i, data in enumerate(progress_bar):
        vessel_depth = data["vessel_depth"]
        liquid_depth = data["liquid_depth"]
        #inputs = torch.cat([vessel_depth, liquid_depth], dim=1)
        #inputs = data["depth_image"]
        targets = torch.stack([data["vol_liquid"], data["vol_vessel"]], dim=1)
        targets = targets.float()

        optimizer.zero_grad()
        outputs = model(vessel_depth, liquid_depth)

        loss = criterion(outputs, targets)
        # Calculate RMSE
        rmse = torch.sqrt(loss).item()
        rmse_epoch += rmse

        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item(), "RMSE": rmse})

    # get the average RMSE for the epoch
    rmse_epoch /= len(train_loader)
    print(f"RMSE for epoch {epoch_str}: {rmse_epoch:.2f}")
    
data_dir = "data/processed"
batch_size = 4
num_epochs = 10
learning_rate = 0.001

# Load the dataset
dataset = VesselCaptureDataset(data_dir)
print(f"Loaded {len(dataset)} samples")

# Split the dataset into training and test data
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Set up the data loader and training parameters for the training data
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
train_size = len(train_data)

# Set up the data loader and training parameters for the test data
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
test_size = len(test_data)

model = VolumeNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_str = "Epoch " + str(epoch + 1)
    train(model, criterion, optimizer, train_loader, epoch_str)

# Save the trained model
torch.save(model.state_dict(), "models/volume_model.pth")

# Evaluate the model on the test data
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        vessel_depth = data["vessel_depth"]
        liquid_depth = data["liquid_depth"]
        #inputs = torch.cat([vessel_depth, liquid_depth], dim=1)
        #inputs = data["depth_image"]
        targets = torch.stack([data["vol_liquid"], data["vol_vessel"]], dim=1)
        targets = targets.float()

        outputs = model(vessel_depth, liquid_depth)

        # first element of output is volume of liquid, second is volume of vessel
        predicted_vol_liquid = outputs[0].item()
        actual_vol_liquid = targets[0].item()
        predicted_vol_vessel = outputs[1].item()
        actual_vol_vessel = targets[1].item()

        # calculate the RMSE for the batch






        loss = criterion(outputs, targets)
        # Calculate RMSE
        rmse = torch.sqrt(loss).item()

        print(f"RMSE for test batch {i + 1}: {rmse:.2f}")

