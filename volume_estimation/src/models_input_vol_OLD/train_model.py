from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from src.data.dataloader import VesselCaptureDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from src.models_input_vol_OLD.model_new import VolumeNet
import time
import statistics
import numpy as np
import matplotlib.pyplot as plt
from src.models_input_vol_OLD.validate_model import validate
import math
#import wandb
#wandb.init(project="volume-estimation")

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # Use GPU if available

print("Device used: ", device)


# Define the training loop
def train(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    epoch_str,
    train_size,
    valid_size,
):
    #wandb.watch(model, criterion, log="all", log_freq=10)
    #log_interval = 10
    
    model.train()

    # Wrap train_loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=epoch_str)

    rmse_epoch = 0
    loss_epoch = 0

    losses_train = []
    losses_valid = []

    for i, data in enumerate(progress_bar):
        #vessel_depth = data["vessel_depth"].to(device)
        #liquid_depth = data["liquid_depth"].to(device)
        input1 = data["vessel_depth"].to(device)
        #print(input1.shape)
        #print(input1.mean())
        #print(input1)
        input2 = data["liquid_depth"].to(device)

        input3 = data["vol_vessel"].to(device)
        #print(input2.shape)
        #print(input2.mean())
        #print(input2)

        # if one of the images is nan, skip this batch
        if torch.isnan(input1).any() or torch.isnan(input2).any():
            continue
        # vessel_depth = data["segmentation_vessel"].to(device)
        # liquid_depth = data["segmentation_liquid"].to(device)

        # vessel_vol = data["vol_vessel"]
        # vessel_vol = vessel_vol.view(vessel_depth.shape[0], 1, 1).repeat(1, 480, 640)
        # vessel_vol = vessel_vol.view(vessel_depth.shape[0], 1, 1).repeat(1, 160, 214)

        # vessel_vol = vessel_vol.to(device)

        # inputs = torch.cat([vessel_depth, liquid_depth], dim=1)
        # inputs = data["depth_image"]
        targets = data["vol_liquid"].to(device)
        targets = targets.float()

        # print("Targets batch ", i, ":", targets)

        optimizer.zero_grad()
        outputs = model(input1, input2, input3)
        #print(outputs.shape)
        #print(outputs)

        loss = criterion(outputs, targets.unsqueeze(1))

        res = outputs - targets.unsqueeze(1)
        # print("Residuals batch ", i, ":", res)
        res_squared = res**2
        # print("Residuals squared batch ", i, ":", res_squared)
        # print("Residuals squared sum batch ", i, ":", torch.sum(res_squared))
        mean_squared_error = torch.sum(res_squared) / batch_size_train
        # print("Mean squared error batch ", i, ":", mean_squared_error)

        # print("Outputs batch ", i, ":", outputs)

        # print("Loss Batch ", i, ":", loss)
        # print("Loss item batch ", i, ":", loss.item())
        # add loss to list
        losses_train.append(loss.item())

        # Calculate RMSE
        # rmse = torch.sqrt(loss).item()
        rmse = torch.sqrt(loss)
        rmse_epoch += rmse.item()
        loss_epoch += loss.item()

        loss.backward()
        optimizer.step()

        #if i % log_interval == 0:
        #    wandb.log({"loss": loss})

        # Update progress bar
        progress_bar.set_postfix(
            {"loss": loss_epoch / (i + 1), "RMSE": rmse_epoch / (i + 1)}
        )

    # get the average RMSE for the epoch
    rmse_epoch /= train_size / batch_size_train
    print(f"RMSE for epoch {epoch_str} on training data: {rmse_epoch:.2f}")

    # model.eval()
    # get loss and rmse for validation set
    loss_valid, rmse_valid = validate(model, valid_loader, valid_size)
    print(f"Loss for epoch {epoch_str} on validation data: {loss_valid:.2f}")
    print(f"RMSE for epoch {epoch_str} on validation data: {rmse_valid:.2f}")

    losses_valid.append(loss_valid)

    return statistics.mean(losses_train), statistics.mean(losses_valid)


data_dir = "data/processed"
#batch_size_train = 16
num_epochs = 200
#learning_rate = 0.0001

learning_rates = [0.001]
batch_sizes = [8]
dropout_rates = [0.2]

best_rmse = float('inf')
best_params = {}
i = 0

for learning_rate in learning_rates:
    for batch_size_train in batch_sizes:
        for dropout_rate in dropout_rates:
            # Initialize the model with the current hyperparameters
            print("Starting training run ", i+1, " of ", learning_rates.__len__() * batch_sizes.__len__() * dropout_rates.__len__(), "\n")

            # random learning rate between 0.0001 and 0.001
            #learning_rate = np.random.uniform(0.0001, 0.001)
            #learning_rate = 0.0001
            print("Learning rate: ", learning_rate)

            # random batch size of 2, 4, 8, 16, 32
            #batch_size_train = int(np.random.choice([2, 4, 8, 16, 32]))
            #batch_size_train = 16
            print("Batch size: ", batch_size_train)

            # random dropout rate between 0.1 and 0.5

            #dropout_rate = np.random.uniform(0.1, 0.5) 
            #dropout_rate = 0.2
            print("Dropout rate: ", dropout_rate)

            # Load the dataset
            dataset = VesselCaptureDataset(data_dir)
            print(f"Loaded {len(dataset)} samples.")

            # Split the dataset into training and test data
            train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)

            train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)

            print(f"Training on {len(train_data)} samples.")
            # print(f"Validating on {len(valid_data)} samples.")
            print(f"Testing on {len(test_data)} samples.")

            # Set up the data loader and training parameters for the training data
            train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)

            # do data augmentation on training data


            train_size = len(train_data)

            # Set up the data loader and training parameters for the validation data
            valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True)
            valid_size = len(valid_data)

            model = VolumeNet(dropout_rate=dropout_rate)
            model = model.to(device)  # Send net to GPU if available
            criterion = nn.MSELoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train the model
            losses_total_train = []
            losses_total_valid = []
            for epoch in range(num_epochs):
                epoch_str = "Epoch " + str(epoch + 1)
                losses_train, losses_valid = train(
                    model,
                    criterion,
                    optimizer,
                    train_loader,
                    valid_loader,
                    epoch_str,
                    train_size,
                    valid_size,
                )
                losses_total_train.append(losses_train)
                losses_total_valid.append(losses_valid)

            loss_valid, rmse_valid = validate(model, valid_loader, valid_size)

            # Check if the current combination of hyperparameters yields the best result
            if rmse_valid < best_rmse:
                best_rmse = rmse_valid
                best_params = {'learning_rate': learning_rate, 'batch_size': batch_size_train, 'dropout_rate': dropout_rate}
                print("New best RMSE: ", best_rmse)
                # Save the trained model
                torch.save(model.state_dict(), "models/volume_model_input_vol_log.pth")

            # save final loss and rmse for this training run to txt file
            with open("results_input_vol_testing.txt", "a") as f:
                f.write("Training run " + str(i+1) + " of 100\n")
                f.write("Learning rate: " + str(learning_rate) + "\n")
                f.write("Batch size: " + str(batch_size_train) + "\n")
                f.write("Dropout rate: " + str(dropout_rate) + "\n")
                # write averag loss of the last 5 epochs
                f.write("Average train RMSE of last 5 epochs: " + str(math.sqrt(statistics.mean(losses_total_train[-5:]))) + "\n")
                f.write("Average valid RMSE of last 5 epochs: " + str(math.sqrt(statistics.mean(losses_total_valid[-5:]))) + "\n")
                f.write("Final valid loss: " + str(loss_valid) + "\n")
                f.write("Final valid RMSE: " + str(rmse_valid) + "\n\n")
    
            i += 1

# write best hyperparameters to txt file
with open("results_input_vol_testing.txt", "a") as f:
    f.write("Best hyperparameters: " + str(best_params) + "\n")
    f.write("Best RMSE: " + str(best_rmse) + "\n\n")

# save losses to txt file
with open("losses_input_vol_testing.txt", "a") as f:
    f.write("Losses train: " + str(losses_total_train) + "\n")
    f.write("Losses valid: " + str(losses_total_valid) + "\n\n")
    

'''
print("Creating figure")
plt.figure(figsize=(10, 5))
# print(losses_total)
plt.plot((losses_total_train))
plt.plot((losses_total_valid))
plt.title("Loss over epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper right")
plt.yscale("log")
plt.show()
plt.savefig(time.strftime("%d%m%Y-%H%M") + ".png")


# Save the trained model
torch.save(model.state_dict(), "models/volume_model.pth")

'''
