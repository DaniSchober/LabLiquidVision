import torch
import numpy as np

def validate(model, valid_loader, valid_size):

    '''
    
        Validate the model

        Args:
            model (VolumeNet): model to validate
            valid_loader (DataLoader): data loader for validation data
            valid_size (int): size of validation set

        Returns:
            loss_liquid (float): loss for liquid volume
            rmse_liquid (float): RMSE for liquid volume

    '''
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

            targets = data["vol_liquid"]
            targets = targets.float()

            outputs = model(vessel_depth, liquid_depth)
            outputs = outputs.squeeze(0)
            targets = targets.squeeze(0)

            # first element of output is volume of liquid, second is volume of vessel
            predicted_vol_liquid = outputs.item()
            actual_vol_liquid = targets.item()

            # calculate squared error for item
            squared_error_liquid = (predicted_vol_liquid - actual_vol_liquid) ** 2

            # add squared error to total
            squared_error_liquid_total += squared_error_liquid

            squared_error_liquid_array = np.append(
                squared_error_liquid_array, squared_error_liquid**0.5
            )

        # calculate RMSE for test set
        rmse_liquid = (squared_error_liquid_total / valid_size) ** 0.5
        loss_liquid = squared_error_liquid_total / valid_size

        return loss_liquid, rmse_liquid
