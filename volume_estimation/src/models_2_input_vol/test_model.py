import torch
from sklearn.model_selection import train_test_split
from src.data.dataloader import VesselCaptureDataset
from torch.utils.data import DataLoader
from src.models_2_input_vol.model import VolumeNet
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sklearn.metrics as metrics
import cv2
import os

# Set font and fontsize globally
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 11

'''

Script to test the model with vessel volume input on the test set


'''
def test(data_dir):

    '''
    
        Test the model with vessel volume input on the test set

        Args:
            data_dir (str): path to data directory

        Prints:
            Size of test set
            RMSE for test set
            Mean percentage error for test set
            R2 score for test set
            Max error
            Min error
            Samples with max error
            Samples with min error
            Results for each vessel in test set

        Shows:
            Histogram of RMSE for test set
            Scatter plot of predicted volume vs actual volume for test set
    
    '''

    # Load the dataset
    dataset = VesselCaptureDataset(data_dir)

    # Split the dataset into training and test data
    train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)

    # Set up the data loader and training parameters for the test data
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_size = len(test_data)

    print("Size test set: ", test_size)

    model = VolumeNet(dropout_rate=0.2)
    model.load_state_dict(torch.load("models/volume_model_with_vol.pth"))
    model.eval()

    squared_error_liquid_total = 0
    squared_error_liquid_array = []
    predicted_vol_liquid_list = []
    actual_vol_liquid_list = []
    vessel_name_list = []

    # create output folder for images with volume estimation

    output_folder = "output_with_vol_input"
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
            vessel_name = vessel_name[0]
            vessel_name_list.append(vessel_name)


            # open image from test set and draw volume estimation on it
            print("Image path: ", data["image_path"][0])
            img = cv2.imread(data["image_path"][0])

            # draw volume estimation on image
            cv2.putText(
                img,
                "Predicted Volume liquid: " + str(int(predicted_vol_liquid)),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img,
                "Real volume liquid: " + str(int(actual_vol_liquid)),
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
            # add squared error to total
            squared_error_liquid_total += squared_error_liquid

            squared_error_liquid_array = np.append(
                squared_error_liquid_array, squared_error_liquid**0.5
            )

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

        # calculate R2 score for test set
        r2_score = metrics.r2_score(actual_vol_liquid_list, predicted_vol_liquid_list)

        # get maximum error
        max_error = max(squared_error_liquid_array)

        # get minimum error
        min_error = min(squared_error_liquid_array)

        # print overall results
        print("\n\n\n")
        print("TOTAL RESULTS:")
        print("Mean of real volume: ", avg_liquid_total)
        print("RMSE: ", rmse_liquid)
        print("MAPE: ", mean_percentage_error * 100, "%")
        print("R2 score: ", r2_score)
        print("Max error: ", max_error)
        print("Samples with max error: ", np.argpartition(squared_error_liquid_array, -3)[-3:]) # print 3 samples with maximum error
        print("Min error: ", min_error)
        print("Samples with min error: ", np.argpartition(squared_error_liquid_array, 3)[:3]) # print 3 samples with minimum error
        print("\n\n\n")

        print("RESULTS FOR EACH VESSEL:")
        print("\n")

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
        vessel_name_list_unique = np.unique(vessel_name_list)
        # Create a colormap object with the desired colormap
        cmap = cm.get_cmap("Paired", len(vessel_name_list_unique))

        for i in range(len(vessel_name_list_unique)):
            # get indices of vessel 
            indices = [
                j for j, x in enumerate(vessel_name_list) if x == vessel_name_list_unique[i]
            ]

            # get predicted and actual volume for vessel 
            predicted_vol_liquid_list_vessel = [
                predicted_vol_liquid_list[j] for j in indices
            ]
            actual_vol_liquid_list_vessel = [actual_vol_liquid_list[j] for j in indices]

            # calculate RMSE for vessel 
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

            # calculate mean liquid volume for vessel 
            avg_liquid_vessel = (
                sum(actual_vol_liquid_list_vessel) / len(actual_vol_liquid_list_vessel)
            )

            # calculate R2 score for vessel 
            r2_score = metrics.r2_score(
                actual_vol_liquid_list_vessel, predicted_vol_liquid_list_vessel
            )

            # calculate mean percentage error for vessel
            mean_percentage_error = (
                sum((((np.array(predicted_vol_liquid_list_vessel)-np.array(actual_vol_liquid_list_vessel))/np.array(actual_vol_liquid_list_vessel))**2)**0.5))/len(actual_vol_liquid_list_vessel)
            
            # get max error
            max_error = max((
                (
                    np.array(predicted_vol_liquid_list_vessel)
                    - np.array(actual_vol_liquid_list_vessel)
                )
                ** 2)**0.5
            )

            # get min error
            min_error = min((
                (
                    np.array(predicted_vol_liquid_list_vessel)
                    - np.array(actual_vol_liquid_list_vessel)
                )
                ** 2)**0.5
            )

            print("Mean liquid volume", vessel_name_list_unique[i], ":", avg_liquid_vessel)
            print("RMSE", vessel_name_list_unique[i], ":", rmse_liquid_vessel)
            print("MAPE", vessel_name_list_unique[i], ":", mean_percentage_error*100, "%")
            print("R2 score", vessel_name_list_unique[i], ":", r2_score)
            print("Max error", vessel_name_list_unique[i], ":", max_error)
            print("Min error", vessel_name_list_unique[i], ":", min_error)
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

        
