from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from src.data.dataloader import VesselCaptureDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from src.models_no_input_vol_testing.model_new import VolumeNet
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Set font and fontsize globally
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 11

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
model = VolumeNet(dropout_rate=0.4)
model.load_state_dict(torch.load("models/volume_model_testing.pth"))
model.eval()

squared_error_liquid_total = 0
squared_error_vessel_total = 0
squared_error_liquid_array = []
predicted_vol_liquid_list = []
actual_vol_liquid_list = []
vessel_name_list = []

# create output folder for images with volume estimation
import os

output_folder = "output_no_vol_input"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


output_folder_res = "output"
if not os.path.exists(output_folder_res):
    os.makedirs(output_folder_res)


with torch.no_grad():
    for i, data in enumerate(test_loader):
        input1 = data["vessel_depth_scaled"]
        input2 = data["liquid_depth_scaled"]
        vessel_name = data["vessel_name"]

        # vessel_depth = data["segmentation_vessel"]
        # liquid_depth = data["segmentation_liquid"]

        # vessel_vol = data["vol_vessel"]
        # vessel_vol = vessel_vol.view(vessel_depth.shape[0], 1, 1).repeat(1, 160, 214)
        # inputs = torch.cat([vessel_depth, liquid_depth], dim=1)
        # inputs = data["depth_image"]
        targets = data["vol_liquid"]
        targets = targets.float()

        # if one of the images is nan, skip this batch
        if torch.isnan(input1).any() or torch.isnan(input2).any():
            continue

        #print("Liquid_depth shape: ", liquid_depth.shape)
        outputs = model(input1, input2)

        outputs = outputs.squeeze(0)
        targets = targets.squeeze(0)
        print("Sample ", i, ":", outputs, targets)

        # first element of output is volume of liquid, second is volume of vessel
        predicted_vol_liquid = outputs.item()
        actual_vol_liquid = targets.item()
        # vessel_name = vessel_name.item()

        # save them in a list
        predicted_vol_liquid_list.append(predicted_vol_liquid)
        actual_vol_liquid_list.append(actual_vol_liquid)
        # convert vessel name from list object to string
        vessel_name = vessel_name[0]

        vessel_name_list.append(vessel_name)
        # predicted_vol_vessel = outputs[1].item()
        # actual_vol_vessel = targets[1].item()

        '''

        # open image from test set and draw volume estimation on it
        import cv2

        # img = cv2.imread(data["image_path"])
        print("Image path: ", data["image_path"][0])
        img = cv2.imread(data["image_path"][0])

        # draw volume estimation on image
        cv2.putText(
            img,
            "Volume liquid: " + str(predicted_vol_liquid),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            img,
            "Actual volume liquid: " + str(actual_vol_liquid),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # save image with volume estimation
        #cv2.imwrite(output_folder + "/image_" + str(i) + ".png", img)
        '''

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
    rmse_liquid = (squared_error_liquid_total / test_size) ** 0.5

    # Plot histogram of root squared errors
    plt.figure(figsize=(6.3, 5))
    plt.hist(squared_error_liquid_array, bins=100)
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.title("Histogram of RMSE for liquid volume estimation on the test set")
    plt.tight_layout()
    plt.savefig(output_folder_res + "/squared_error_liquid_no_input_vol_testing.png", format="png", dpi=600
                )
    plt.show()

    # plot predicted volume vs actual volume in scatter plot with color depending on vessel name
    # get unique vessel names
    vessel_name_list_unique = np.unique(vessel_name_list)
    # Create a colormap object with the desired colormap
    cmap = cm.get_cmap("Paired", len(vessel_name_list_unique))

    plt.figure(figsize=(6.3, 5))

    # print(vessel_name_list)
    # plot scatter plot
    for i in range(len(vessel_name_list_unique)):
        # get indices of vessel name
        indices = [
            j for j, x in enumerate(vessel_name_list) if x == vessel_name_list_unique[i]
        ]

        # get predicted and actual volume for vessel name
        predicted_vol_liquid_list_vessel = [
            predicted_vol_liquid_list[j] for j in indices
        ]
        actual_vol_liquid_list_vessel = [actual_vol_liquid_list[j] for j in indices]

        # calculate RMSE for vessel name
        rmse_liquid_vessel = (
            sum(
                (
                    np.array(predicted_vol_liquid_list_vessel)
                    - np.array(actual_vol_liquid_list_vessel)
                )
                ** 2
            )
            / len(predicted_vol_liquid_list_vessel)
        ) ** 0.5
        print("RMSE", vessel_name_list_unique[i], ":", rmse_liquid_vessel)
        print("Average actual volume", vessel_name_list_unique[i], ": ", sum(actual_vol_liquid_list_vessel)/len(actual_vol_liquid_list_vessel))
        # plot scatter plot

        plt.scatter(
            actual_vol_liquid_list_vessel,
            predicted_vol_liquid_list_vessel,
            label=vessel_name_list_unique[i],
            color=cmap(i),
        )

    # xlabel
    plt.xlabel("Actual volume (mL)")
    # ylabel
    plt.ylabel("Predicted volume (mL)")
    # plot diagonal line
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c=".3")
    # save scatter plot
    plt.legend()
    plt.title("Scatter plot of predicted vs actual volume")
    # make legend smaller
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_folder_res + "/scatter_plot_no_input_vol_testing.png", format="png", dpi=600)
    # show scatter plot
    plt.show()

    print("RMSE liquid: ", rmse_liquid)
    # print("RMSE vessel: ", rmse_vessel)
