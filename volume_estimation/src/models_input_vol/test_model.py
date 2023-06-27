from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from src.data.dataloader import VesselCaptureDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from src.models_input_vol_testing.model_new import VolumeNet
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sklearn.metrics as metrics

# Set font and fontsize globally
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 11

data_dir = "data/processed"

# Load the dataset
dataset = VesselCaptureDataset(data_dir)
print(f"Loaded {len(dataset)} samples")

# Split the dataset into training and test data
train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)


# Set up the data loader and training parameters for the test data
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
test_size = len(test_data)

print("Size test set: ", test_size)

# load model from models/model_new.py
model = VolumeNet(dropout_rate=0.05)
model.load_state_dict(torch.load("models/volume_model_input_vol_log_final.pth"))
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
        input1 = data["vessel_depth"]
        input2 = data["liquid_depth"]
        input3 = data["vol_vessel"]
        vessel_name = data["vessel_name"]

        # vessel_depth = data["segmentation_vessel"]
        # liquid_depth = data["segmentation_liquid"]

        # vessel_vol = data["vol_vessel"]
        # vessel_vol = vessel_vol.view(vessel_depth.shape[0], 1, 1).repeat(1, 160, 214)
        # inputs = torch.cat([vessel_depth, liquid_depth], dim=1)
        # inputs = data["depth_image"]
        targets = data["vol_liquid"]
        targets = targets.float()

        #print("Liquid_depth shape: ", liquid_depth.shape)
        outputs = model(input1, input2, input3)

        outputs = outputs.squeeze(0)
        targets = targets.squeeze(0)
        print("Sample ", i, ":", outputs, targets)

        predicted_vol_liquid = outputs.item()
        actual_vol_liquid = targets.item()

        # save them in a list
        predicted_vol_liquid_list.append(predicted_vol_liquid)
        actual_vol_liquid_list.append(actual_vol_liquid)
        # convert vessel name from list object to string
        vessel_name = vessel_name[0]

        vessel_name_list.append(vessel_name)
        # predicted_vol_vessel = outputs[1].item()
        # actual_vol_vessel = targets[1].item()

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
        cv2.imwrite(output_folder + "/image_" + str(i) + ".png", img)

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



    # calculate mean percentage error for test set
    mean_percentage_error = (
        sum(
            (
                (
                    (np.array(predicted_vol_liquid_list) - np.array(actual_vol_liquid_list))
                    / np.array(actual_vol_liquid_list)
                )
                ** 2
            )**0.5
        )
    ) / len(actual_vol_liquid_list)

    # calculate average of liquid amount
    avg_liquid_total = sum(actual_vol_liquid_list) / len(actual_vol_liquid_list)

    print("\n\n\n")
    print("Mean of actual volume: ", avg_liquid_total)
    print("RMSE liquid: ", rmse_liquid)
    #print("RMSE liquid percentage: ", rmse_liquid / avg_liquid_total * 100, "%")
    print("Mean percentage error test set: ", mean_percentage_error * 100, "%")

    # calculate R2 score for test set
    r2_score = metrics.r2_score(actual_vol_liquid_list, predicted_vol_liquid_list)
    print("R2 score test set: ", r2_score)

    # get maximum error
    max_error = max(squared_error_liquid_array)
    print("Max error: ", max_error)

    # print 3 samples with maximum error
    print("Samples with max error: ", np.argpartition(squared_error_liquid_array, -5)[-5:])

    # get minimum error
    min_error = min(squared_error_liquid_array)
    print("Min error: ", min_error)

    # print 3 samples with minimum error
    print("Samples with min error: ", np.argpartition(squared_error_liquid_array, 5)[:5])

    percentage_error = (
                (
                    (np.array(predicted_vol_liquid_list) - np.array(actual_vol_liquid_list))
                    / np.array(actual_vol_liquid_list)
                )
                ** 2
            )**0.5
    
    # print samples with highest percentage error
    print("Samples with highest percentage error: ", np.argpartition(percentage_error, -5)[-5:])
    print("\n\n\n")

    # Plot histogram of root squared errors
    plt.figure(figsize=(6.3, 3))
    plt.hist(squared_error_liquid_array, bins=100)
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.title("RMSE for Volume Estimation on Test Set (with Vessel Volume Input)")
    plt.tight_layout()
    plt.savefig(output_folder_res + "/squared_error_liquid_vol_input.png", format="png", dpi=300)
    plt.show()

    # plot predicted volume vs actual volume in scatter plot with color depending on vessel name
    # get unique vessel names
    vessel_name_list_unique = np.unique(vessel_name_list)
    # Create a colormap object with the desired colormap
    cmap = cm.get_cmap("Paired", len(vessel_name_list_unique))

    plt.figure(figsize=(6.3, 4.5))

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

        # calculate mean liquid volume for vessel name
        avg_liquid_vessel = (
            sum(actual_vol_liquid_list_vessel) / len(actual_vol_liquid_list_vessel)
        )
        print("Mean liquid volume", vessel_name_list_unique[i], ":", avg_liquid_vessel)

        # calculate R2 score for vessel name
        r2_score = metrics.r2_score(
            actual_vol_liquid_list_vessel, predicted_vol_liquid_list_vessel
        )
        print("R2 score", vessel_name_list_unique[i], ":", r2_score)

        # calculate mean percentage error for vessel name
        mean_percentage_error = (
            sum((((np.array(predicted_vol_liquid_list_vessel)-np.array(actual_vol_liquid_list_vessel))/np.array(actual_vol_liquid_list_vessel))**2)**0.5))/len(actual_vol_liquid_list_vessel)
        
        print("Mean percentage error", vessel_name_list_unique[i], ":", mean_percentage_error*100, "%")

        # get max error
        max_error = max((
            (
                np.array(predicted_vol_liquid_list_vessel)
                - np.array(actual_vol_liquid_list_vessel)
            )
            ** 2)**0.5
        )

        print("Max error", vessel_name_list_unique[i], ":", max_error)

        # get min error
        min_error = min((
            (
                np.array(predicted_vol_liquid_list_vessel)
                - np.array(actual_vol_liquid_list_vessel)
            )
            ** 2)**0.5
        )

        print("Min error", vessel_name_list_unique[i], ":", min_error)
        
        avg_liquid = sum(actual_vol_liquid_list_vessel) / len(actual_vol_liquid_list_vessel)
        # calculate percentage of error 
        percentage_error = rmse_liquid_vessel / avg_liquid * 100
        #print("Mean percentage error", vessel_name_list_unique[i], ":", percentage_error, "%")
        # plot scatter plot

        print("\n")

        plt.scatter(
            actual_vol_liquid_list_vessel,
            predicted_vol_liquid_list_vessel,
            label=vessel_name_list_unique[i],
            color=cmap(i),
        )

    # xlabel
    plt.xlabel("Real Volume (mL)")
    # ylabel
    plt.ylabel("Predicted Volume (mL)")
    # plot diagonal line
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c=".3")
    # save scatter plot
    plt.legend()
    plt.title("Predicted vs. Real Volume on Test Set (with Vessel Volume Input)")
    # make legend smaller
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_folder_res + "/scatter_plot_input_vol.png", format="png", dpi=600)
    # show scatter plot
    plt.show()

    
