from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from src.data.dataloader import VesselCaptureDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from src.models_input_vessel_vol.model_new import VolumeNet

data_dir = "data/processed"

# Load the dataset
dataset = VesselCaptureDataset(data_dir)
print(f"Loaded {len(dataset)} samples")

# Split the dataset into training and test data
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)



# Set up the data loader and training parameters for the test data
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
test_size = len(test_data)

print("Size test set: ", test_size)

# load model from models/model_new.py
model = VolumeNet()
model.load_state_dict(torch.load("models/volume_model.pth"))
model.eval()

squared_error_liquid_total = 0
squared_error_vessel_total = 0
squared_error_liquid_array = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        vessel_depth = data["vessel_depth"]
        liquid_depth = data["liquid_depth"]
        vessel_vol = data["vol_vessel"]
        vessel_vol = vessel_vol.view(vessel_depth.shape[0], 1, 1).repeat(1, 480, 640)
        #inputs = torch.cat([vessel_depth, liquid_depth], dim=1)
        #inputs = data["depth_image"]
        targets = data["vol_liquid"]
        targets = targets.float()

        outputs = model(vessel_depth, liquid_depth, vessel_vol)
        
        outputs = outputs.squeeze(0)
        targets = targets.squeeze(0)
        print("Sample ", i, ":", outputs, targets)

        # first element of output is volume of liquid, second is volume of vessel
        predicted_vol_liquid = outputs.item()
        actual_vol_liquid = targets.item()
        #predicted_vol_vessel = outputs[1].item()
        #actual_vol_vessel = targets[1].item()

        # calculate squared error for item
        squared_error_liquid = (predicted_vol_liquid - actual_vol_liquid) ** 2
        #squared_error_vessel = (predicted_vol_vessel - actual_vol_vessel) ** 2

        # add squared error to total
        squared_error_liquid_total += squared_error_liquid

        squared_error_liquid_array = squared_error_liquid_array.append(squared_error_liquid)
        #squared_error_vessel_total += squared_error_vessel
    
    # calculate RMSE for test set
    rmse_liquid = (squared_error_liquid_total / test_size) ** 0.5
    #rmse_vessel = (squared_error_vessel_total / test_size) ** 0.5

    # plot histogram of squared errors
    import matplotlib.pyplot as plt
    plt.hist(squared_error_liquid_array)
    plt.show()
    # save histogram of squared errors
    plt.savefig("squared_error_liquid.png")
    

    print("RMSE liquid: ", rmse_liquid)
    #print("RMSE vessel: ", rmse_vessel)







