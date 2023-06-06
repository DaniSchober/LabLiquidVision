import torch
from src.models.model import VesselNet
import numpy as np
import sys
import os
import warnings
import cv2

# add parent folder to sys.path
sys.path.insert(0, os.path.abspath(".."))

# ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ignore RuntimeWarning messages
warnings.filterwarnings("ignore", category=RuntimeWarning)

import segmentation_and_depth.src.visualization.visualize as vis
import segmentation_and_depth.src.models.model as NET_FCN  # The net Class


def predict(image_path):
    model_path = r"../segmentation_and_depth/models/55__03042023-2211.torch"  # Trained model path
    UseGPU = True  # Use GPU or not
    MaxSize = 3000

    # get depth maps from segmentation and depth model
    DepthList = ["EmptyVessel_Depth", "ContentDepth", "VesselOpening_Depth"]
    MaskList = [
        "VesselMask",
        "ContentMaskClean",
        "VesselOpeningMask",
    ]  # List of segmentation Masks to predict
    depth2Mask = {
        "EmptyVessel_Depth": "VesselMask",
        "ContentDepth": "ContentMaskClean",
        "VesselOpening_Depth": "VesselOpeningMask",
    }  # Connect depth map to segmentation mask

    model = NET_FCN.Net(MaskList=MaskList, DepthList=DepthList)
    # Load model weights depending on GPU usage
    if UseGPU == True:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Load image
    image = cv2.imread(image_path)
    image = vis.ResizeToMaxSize(image, MaxSize)
    image = np.expand_dims(image, axis=0)

    # Run inference
    with torch.no_grad():
        PrdDepth, PrdProb, PrdMask = model.forward(
            Images=image, TrainMode=False, UseGPU=UseGPU
        )  # Run net inference and get prediction

    model = VesselNet()
    model.load_state_dict(torch.load("models/vessel_net.pth"))

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
