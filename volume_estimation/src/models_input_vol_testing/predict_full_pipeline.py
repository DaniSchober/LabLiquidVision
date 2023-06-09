import torch

import numpy as np
import sys
import os
import warnings
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F


# add parent folder to sys.path
sys.path.insert(0, os.path.abspath(".."))

# ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ignore RuntimeWarning messages
warnings.filterwarnings("ignore", category=RuntimeWarning)

import segmentation_and_depth.src.visualization.visualize as vis
import segmentation_and_depth.src.models.model as NET_FCN  # The net Class
from volume_estimation.src.models_input_vol_testing.model_new import VolumeNet


def predict(image_path, predict_volume=False, save_segmentation=False, save_depth=False):
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
    image_new = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)

    # Run inference
    with torch.no_grad():
        PrdDepth, PrdProb, PrdMask = model.forward(
            Images=image, TrainMode=False, UseGPU=UseGPU
        )  # Run net inference and get prediction

    with torch.no_grad():
        PrdDepth, PrdProb, PrdMask = model.forward(
            Images=image, TrainMode=False, UseGPU=UseGPU
        )

    Prd = {}
    for nm in PrdDepth:
        # convert to numpy and transpose to (H,W,C)
        Prd[nm] = (PrdDepth[nm].transpose(1, 2).transpose(2, 3)).data.cpu().numpy()
    for nm in PrdMask:
        # convert to numpy
        Prd[nm] = (PrdMask[nm]).data.cpu().numpy()

    # loop over all masks and depth maps
    for nm in Prd:
        if save_segmentation == True:
            if nm in MaskList:
                np.save(
                    image_path.replace(".png", "_" + nm + ".npy"),
                    Prd[nm],
                )
                # copy mask for visualization
                tmIm = Prd[nm][0].copy()
                # normalize mask to values between 0-255
                if (
                    Prd[nm][0].max() > 255
                    or Prd[nm][0].min() < 0
                    or np.ndim(Prd[nm][0]) == 2
                ):
                    if tmIm.max() > tmIm.min():  #
                        tmIm[tmIm > 1000] = 0
                        tmIm = tmIm - tmIm.min()
                        tmIm = tmIm / tmIm.max() * 255
                    if np.ndim(tmIm) == 2:
                        tmIm = cv2.cvtColor(
                            tmIm.astype(np.uint8), cv2.COLOR_GRAY2BGR
                        )

                image = Prd[nm].squeeze()
                image = image / image.max()
                image = image * 255
                image = image.astype(np.uint8)
                cv2.imwrite(
                    image_path.replace(".png", nm + ".png"), image
                )
        

        if nm in DepthList:
            # copy depth map for visualization
            tmIm = Prd[nm].copy()
            tmIm = tmIm.squeeze()
            # np.save(
            #    image_path.replace(".png", nm + ".npy"), tmIm
            # )

            if nm == "ContentDepth":
                tmIm[Prd[depth2Mask[nm]][0] == 0] = 0
                liquid_depth = tmIm
            elif nm == "EmptyVessel_Depth":
                tmIm[Prd[depth2Mask[nm]][0] == 0] = 0
                vessel_depth = tmIm

            if save_depth == True:
                if nm in depth2Mask:
                    # Remove region out side of the object mask from the depth mask
                    tmIm[Prd[depth2Mask[nm]][0] == 0] = 0
                    np.save(
                        image_path.replace(
                            ".png", nm + "_segmented.npy"
                        ),
                        tmIm,
                    )

                    # normalize the rest of the depth map to values between 0-255
                    tmIm[Prd[depth2Mask[nm]][0] == 1] += 10
                    # if mask exists:
                    if tmIm[Prd[depth2Mask[nm]][0] == 1] != []:
                        min = tmIm[Prd[depth2Mask[nm]][0] == 1].min()
                        max = tmIm[Prd[depth2Mask[nm]][0] == 1].max()
                        tmIm[Prd[depth2Mask[nm]][0] == 1] = (
                            tmIm[Prd[depth2Mask[nm]][0] == 1] - min
                        )
                        max = tmIm[Prd[depth2Mask[nm]][0] == 1].max()
                        tmIm[Prd[depth2Mask[nm]][0] == 1] = (
                            tmIm[Prd[depth2Mask[nm]][0] == 1] / max * 255
                        )
                        image = tmIm.squeeze()
                        image = image.astype(np.uint8)
                        # save image
                        cv2.imwrite(
                            image_path.replace(
                                ".png", nm + "_segmented.png"
                            ),
                            image,
                        )
    
    if predict_volume == True:
        liquid_depth = torch.from_numpy(liquid_depth).float()
        # print(liquid_depth.shape)
        # liquid_depth = liquid_depth.unsqueeze(0).unsqueeze(0)
        # print(liquid_depth.shape)
        liquid_depth = F.interpolate(
            liquid_depth.unsqueeze(0).unsqueeze(0),
            size=(160, 214),
            mode="bilinear",
            align_corners=False,
        )
        liquid_depth = liquid_depth.squeeze(0)
        # print(liquid_depth.shape)

        vessel_depth = torch.from_numpy(vessel_depth).float()
        vessel_depth = F.interpolate(
            vessel_depth.unsqueeze(0).unsqueeze(0),
            size=(160, 214),
            mode="bilinear",
            align_corners=False,
        )
        vessel_depth = vessel_depth.squeeze(0)
        # print(vessel_depth.shape)

        model = VolumeNet(dropout_rate=0.2)

        # print current location
        print(os.getcwd())

        model_path_volume = (
            r"../volume_estimation/models/volume_model_no_input_vol_good.pth"  # Trained model path
        )
        model.load_state_dict(torch.load(model_path_volume))

        with torch.no_grad():
            model.eval()
            outputs = model(vessel_depth, liquid_depth)

        print(f"Predicted liquid volume: {outputs[0].item():.2f} mL")

        # save predicted volume to file
        with open(image_path.replace(".png", "volume.txt"), "w") as f:
            print(f"Predicted liquid volume: {outputs[0].item():.2f} mL", file=f)

        """
        Visualize results    
        """
        Prd = {}
        for nm in PrdDepth:
            # convert to numpy and transpose to (H,W,C)
            Prd[nm] = (PrdDepth[nm].transpose(1, 2).transpose(2, 3)).data.cpu().numpy()
        for nm in PrdMask:
            # convert to numpy
            Prd[nm] = (PrdMask[nm]).data.cpu().numpy()

        DepthList = ["EmptyVessel_Depth", "ContentDepth"]
        MaskList = [
            "VesselMask",
            "ContentMaskClean",
        ]  # List of segmentation Masks to predict

        # Visualize results
        count_vis = 1
        # create subplot to show all images in one window
        plt.figure(figsize=(20, 5))
        # show original image
        plt.subplot(1, 4, 1)
        plt.imshow(image_new)
        plt.title("Original Image")

        plt.axis("off")
        # loop over all masks and depth maps
        for nm in Prd:
            """
            if nm in MaskList:
                #count_vis += 1
                # copy mask for visualization
                tmIm = Prd[nm][0].copy()
                # normalize mask to values between 0-255
                if (
                    Prd[nm][0].max() > 255
                    or Prd[nm][0].min() < 0
                    or np.ndim(Prd[nm][0]) == 2
                ):
                    if tmIm.max() > tmIm.min():  #
                        tmIm[tmIm > 1000] = 0
                        tmIm = tmIm - tmIm.min()
                        tmIm = tmIm / tmIm.max() * 255
                    if np.ndim(tmIm) == 2:
                        tmIm = cv2.cvtColor(tmIm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                #plt.subplot(3, 2, count_vis)
                #plt.imshow(tmIm)
                #plt.axis("off")
                #plt.title(nm)
            """
            if nm in DepthList:
                count_vis += 1
                # copy depth map for visualization
                tmIm = Prd[nm].copy()
                # squeeze depth map
                tmIm = tmIm.squeeze()
                if nm in depth2Mask:
                    # Remove region out side of the object mask from the depth mask
                    tmIm[Prd[depth2Mask[nm]][0] == 0] = 0
                    # normalize the rest of the depth map to values between 0-255
                    tmIm[Prd[depth2Mask[nm]][0] == 1] += 10
                    # if mask exists:
                    if tmIm[Prd[depth2Mask[nm]][0] == 1] != []:
                        min = tmIm[Prd[depth2Mask[nm]][0] == 1].min()
                        max = tmIm[Prd[depth2Mask[nm]][0] == 1].max()
                        tmIm[Prd[depth2Mask[nm]][0] == 1] = (
                            tmIm[Prd[depth2Mask[nm]][0] == 1] - min
                        )
                        max = tmIm[Prd[depth2Mask[nm]][0] == 1].max()
                        tmIm[Prd[depth2Mask[nm]][0] == 1] = (
                            tmIm[Prd[depth2Mask[nm]][0] == 1] / max * 255
                        )

                # visualize depth map
                plt.subplot(1, 4, count_vis)
                plt.imshow(tmIm, cmap="CMRmap")
                # turn off axis
                plt.axis("off")
                plt.title(nm)

        # draw text on top of image to show predicted volume with 2 digits after decimal point
        plt.subplot(1, 4, 4)
        plt.axis("off")
        # write text on subplot
        plt.text(
            0.5,
            0.5,
            f"Predicted liquid volume: {outputs[0].item():.2f} ml",
            ha="center",
            va="center",
            fontsize=12,
        )
        # show image

        plt.savefig(image_path.replace(".png", "visualize.png"))

        plt.show()
