import torch
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
            input1 = data["vessel_depth"].to(device)
            input2 = data["liquid_depth"].to(device)
            input3 = data["vol_vessel"].to(device)

            targets = data["vol_liquid"].to(device)
            targets = targets.float()

            # if one of the images is nan, skip this batch
            if torch.isnan(input1).any() or torch.isnan(input2).any():
                continue

            outputs = model(input1, input2, input3)

            outputs = outputs.squeeze(0)
            targets = targets.squeeze(0)

            # first element of output is volume of liquid, second is volume of vessel
            predicted_vol_liquid = outputs.item()
            actual_vol_liquid = targets.item()

            # calculate squared error for item
            squared_error_liquid = (predicted_vol_liquid - actual_vol_liquid) ** 2
            # squared_error_vessel = (predicted_vol_vessel - actual_vol_vessel) ** 2

            # add squared error to total
            squared_error_liquid_total += squared_error_liquid

            squared_error_liquid_array = np.append(
                squared_error_liquid_array, squared_error_liquid**0.5
            )
            # squared_error_vessel_total += squared_error_vessel

        # calculate RMSE for test set
        rmse_liquid = (squared_error_liquid_total / valid_size) ** 0.5
        loss_liquid = squared_error_liquid_total / valid_size

        return loss_liquid, rmse_liquid
