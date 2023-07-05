import torch
from src.models_2_input_vol.model import VolumeNet
import numpy as np
import os
import warnings
import cv2
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F


def predict(folder_path):
    """

    Predict the volume of liquid in a vessel from a random folder of depth maps

    Args:
        folder_path (str): path to folder of depth maps

    Prints:
        Path to random subfolder
        True volume of liquid
        Predicted volume of liquid

    Shows:
        Image with predicted and true volume of liquid

    """

    # load model
    model = VolumeNet()
    model.load_state_dict(torch.load("models/volume_model_with_vol.pth"))
    model.eval()

    # take random folder from data/processed
    folder_path = "data/processed"  # Replace with the path to your folder
    subfolders = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]

    # Select a random subfolder
    random_subfolder = random.choice(subfolders)
    folder_path = random_subfolder
    print("Random subfolder:", random_subfolder)

    # load images from folder
    vessel_depth_path = os.path.join(
        folder_path, "Input_EmptyVessel_Depth_segmented.npy"
    )
    liquid_depth_path = os.path.join(folder_path, "Input_ContentDepth_segmented.npy")
    true_vol_liquid_path = os.path.join(folder_path, "Input_vol_liquid.txt")

    true_vol_vessel_path = os.path.join(folder_path, "Input_vol_vessel.txt")

    if not os.path.exists(vessel_depth_path):
        warnings.warn(f"Could not find vessel depth image at {vessel_depth_path}")
        return
    if not os.path.exists(liquid_depth_path):
        warnings.warn(f"Could not find liquid depth image at {liquid_depth_path}")
        return
    if not os.path.exists(true_vol_liquid_path):
        warnings.warn(f"Could not find liquid volume at {true_vol_liquid_path}")
        return
    if not os.path.exists(true_vol_vessel_path):
        warnings.warn(f"Could not find vessel volume at {true_vol_vessel_path}")
        return

    true_vol_liquid = int(open(true_vol_liquid_path, "r").read().strip())
    true_vol_vessel = int(open(true_vol_vessel_path, "r").read().strip())

    vessel_depth = np.load(vessel_depth_path).astype(np.float32)
    liquid_depth = np.load(liquid_depth_path).astype(np.float32)

    liquid_depth = torch.from_numpy(liquid_depth).float()
    # convert from log to linear space
    liquid_depth = torch.exp(liquid_depth)
    liquid_depth = F.interpolate(
        liquid_depth.unsqueeze(0).unsqueeze(0),
        size=(160, 214),
        mode="bilinear",
        align_corners=False,
    )
    liquid_depth = liquid_depth.squeeze(0)

    vessel_depth = torch.from_numpy(vessel_depth).float()
    # convert from log to linear space
    vessel_depth = torch.exp(vessel_depth)
    vessel_depth = F.interpolate(
        vessel_depth.unsqueeze(0).unsqueeze(0),
        size=(160, 214),
        mode="bilinear",
        align_corners=False,
    )
    vessel_depth = vessel_depth.squeeze(0)

    # Run inference
    with torch.no_grad():
        vol = model.forward(
            vessel_depth, liquid_depth, torch.tensor(true_vol_vessel).unsqueeze(0)
        )  # Run net inference and get prediction

    print("True liquid volume: {} ml".format(true_vol_liquid))
    print(f"Predicted liquid volume: {int(vol[0][0].item())} ml")

    visualize_path = os.path.join(folder_path, "Input_visualize.png")

    if not os.path.exists(visualize_path):
        warnings.warn(f"Could not find visualize image at {visualize_path}")
        return

    # load image
    visualize = cv2.imread(visualize_path)
    plt.imshow(cv2.cvtColor(visualize, cv2.COLOR_BGR2RGB))
    plt.grid(False)
    plt.axis("off")
    # add results to image
    plt.text(
        920,
        150,
        "True liquid volume: {} ml".format(true_vol_liquid),
        color="black",
        fontsize=14,
    )
    plt.text(
        920,
        190,
        "Predicted liquid volume: {} ml".format(int(vol[0][0].item())),
        color="black",
        fontsize=14,
    )
    # open new window full screen
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")

    plt.show()
