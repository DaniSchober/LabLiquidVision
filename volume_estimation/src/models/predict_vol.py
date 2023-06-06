import torch
from src.models.model_new import VolumeNet
import numpy as np
import sys
import os
import warnings
import cv2
import matplotlib.pyplot as plt
import random


def predict(folder_path):
    model = VolumeNet()
    model.load_state_dict(torch.load("models/vessel_net.pth"))
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

    vessel_depth = torch.from_numpy(vessel_depth)
    # print(vessel_depth.shape)
    # add batch dimension
    vessel_depth = vessel_depth.unsqueeze(0)
    # print(vessel_depth.shape)

    # vessel_depth = vessel_depth.permute(0, 2, 1, 3)
    liquid_depth = torch.from_numpy(liquid_depth)
    # add batch dimension
    liquid_depth = liquid_depth.unsqueeze(0)
    # print(liquid_depth.shape)
    # liquid_depth = liquid_depth.permute(0, 2, 1, 3)
    # Run inference
    with torch.no_grad():
        vol = model.forward(
            vessel_depth, liquid_depth
        )  # Run net inference and get prediction

    # print(vol)
    print("True liquid volume: {} ml".format(true_vol_liquid))
    print("True vessel volume: {} ml".format(true_vol_vessel))
    print(f"Predicted liquid volume: {int(vol[0][0].item())} ml")
    print(f"Predicted vessel volume: {int(vol[0][1].item())} ml")

    # open Input_visualize.png
    visualize_path = os.path.join(folder_path, "Input_visualize.png")

    if not os.path.exists(visualize_path):
        warnings.warn(f"Could not find visualize image at {visualize_path}")
        return

    # load image
    visualize = cv2.imread(visualize_path)
    # show image using matplotlib
    # plt.imshow(visualize)
    plt.imshow(cv2.cvtColor(visualize, cv2.COLOR_BGR2RGB))
    # no grid
    plt.grid(False)
    # no axis
    plt.axis("off")
    # add text
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
        "True vessel volume: {} ml".format(true_vol_vessel),
        color="black",
        fontsize=14,
    )
    plt.text(
        920,
        230,
        "Predicted liquid volume: {} ml".format(int(vol[0][0].item())),
        color="black",
        fontsize=14,
    )
    plt.text(
        920,
        270,
        "Predicted vessel volume: {} ml".format(int(vol[0][1].item())),
        color="black",
        fontsize=14,
    )
    # open new window full screen
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    # show image as RGB

    plt.show()

    # add text
    # cv2.putText(visualize, "True liquid volume: {} ml".format(true_vol_liquid), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # cv2.putText(visualize, "True vessel volume: {} ml".format(true_vol_vessel), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # cv2.putText(visualize, "Predicted liquid volume: {} ml".format(int(vol[0][0].item())), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # cv2.putText(visualize, "Predicted vessel volume: {} ml".format(int(vol[0][1].item())), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # cv2.imshow("Visualize", visualize)
    # cv2.waitKey(0)

    """
    depth_image = np.load(args.depth_image).astype(np.float32)
    # print(depth_image.shape)
    depth_image = torch.from_numpy(depth_image)
    # print(depth_image.shape)
    depth_image = depth_image.unsqueeze(0)
    # print(depth_image.shape)


    with torch.no_grad():
        model.eval()
        outputs = model(depth_image)
        vol_liquid = outputs[0][0].item()
        vol_vessel = outputs[0][1].item()

    print(f"Predicted liquid volume: {vol_liquid:.2f} ml")
    print(f"Predicted vessel volume: {vol_vessel:.2f} ml")

    """
