import torch
from src.data.dataloader import VesselCaptureDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from src.models_no_input_vol.model_new import VolumeNet
import numpy as np


def validate(model, valid_loader, valid_size):

    model.eval()

    squared_error_liquid_total = 0
    squared_error_liquid_array = []

    device = (  # Use GPU if available
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            vessel_depth = data["vessel_depth"].to(device)
            liquid_depth = data["liquid_depth"].to(device)

            #vessel_depth = data["segmentation_vessel"].to(device)
            #liquid_depth = data["segmentation_liquid"].to(device)
            #vessel_vol = data["vol_vessel"]
            #vessel_vol = vessel_vol.view(vessel_depth.shape[0], 1, 1).repeat(1, 160, 214)
            #vessel_vol = vessel_vol.to(device)
            #inputs = torch.cat([vessel_depth, liquid_depth], dim=1)
            #inputs = data["depth_image"]
            targets = data["vol_liquid"]
            targets = targets.float()

            outputs = model(vessel_depth, liquid_depth)
            
            outputs = outputs.squeeze(0)
            targets = targets.squeeze(0)
            #print("Sample ", i, ":", outputs, targets)

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

            squared_error_liquid_array = np.append(squared_error_liquid_array, squared_error_liquid**0.5)
            #squared_error_vessel_total += squared_error_vessel
        
        # calculate RMSE for test set
        rmse_liquid = (squared_error_liquid_total / valid_size) ** 0.5
        loss_liquid = (squared_error_liquid_total / valid_size) 

        return loss_liquid, rmse_liquid 







