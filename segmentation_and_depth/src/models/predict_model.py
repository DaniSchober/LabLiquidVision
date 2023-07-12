# Run inference on image with trained model

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import time
import os
import sys
import warnings

# add parent folder to sys.path
sys.path.insert(0, os.path.abspath(".."))

# ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ignore RuntimeWarning messages
warnings.filterwarnings("ignore", category=RuntimeWarning)

import segmentation_and_depth.src.models.model as NET_FCN
import segmentation_and_depth.src.visualization.visualize as vis


"""
This file contains the function to predict the segmentation and depth maps of an input RGB image

The function takes in the following arguments:
    --model_path: path to trained model
    --image_path: path to image to predict

Output:
    Image with predicted segmentation and depth maps saved in example/results
    Single images of predicted segmentation and depth maps saved in example/results

"""


def predict(model_path, image_path):
    # input parameters
    UseGPU = False
    MaxSize = 1000  
    # define masks and depth maps to predict
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

    # Load model
    model = NET_FCN.Net(MaskList=MaskList, DepthList=DepthList)
    # Load model weights depending on GPU usage
    if UseGPU == True:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # print model architecture
    # print(model)
    # save model architecture to file
    with open("output/model.txt", "w") as f:
        print(model, file=f)

    # Load image
    image = cv2.imread(image_path)
    image = vis.ResizeToMaxSize(image, MaxSize)
    image = np.expand_dims(image, axis=0)

    # Run inference
    with torch.no_grad():
        PrdDepth, PrdProb, PrdMask = model.forward(
            Images=image, TrainMode=False, UseGPU=UseGPU
        )  # Run net inference and get prediction

    Prd = {}
    for nm in PrdDepth:
        # convert to numpy and transpose to (H,W,C)
        Prd[nm] = (PrdDepth[nm].transpose(1, 2).transpose(2, 3)).data.cpu().numpy()
        # print("Predicted Depth Map: ", Prd[nm].shape)

        # convert depth map from log space to linear space
        Prd[nm] = np.exp(Prd[nm])

    for nm in PrdMask:
        # convert to numpy
        Prd[nm] = (PrdMask[nm]).data.cpu().numpy()

    print("Predicted Masks and Depth Maps: ", Prd.keys())

    # create folder to save results
    os.makedirs("example/results/" + time.strftime("%d%m%Y-%H%M%S"), exist_ok=True)

    # save path
    save_path = "example/results/" + time.strftime("%d%m%Y-%H%M%S") + "/"

    # Visualize summary of results
    count = 3
    # create subplot to show all images in one window
    plt.figure(figsize=(15, 10))
    # show original image
    plt.subplot(3, 3, 2)
    image_new = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
    plt.imshow(image_new)
    plt.title("Original Image")
    plt.axis("off")
    # loop over all masks and depth maps
    for nm in Prd:
        count += 1
        if nm in MaskList:
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
            # print(tmIm.shape)
            plt.subplot(3, 3, count)
            plt.imshow(tmIm)
            plt.axis("off")
            plt.title(nm)
        else:
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
            plt.subplot(3, 3, count)
            plt.imshow(tmIm, cmap="CMRmap")
            # turn off axis
            plt.axis("off")
            plt.title(nm)
            # save image

    plt.savefig(save_path + "Summary" + ".png", dpi=300)
    plt.show()

    # save single results as images

    image_new = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
    plt.imshow(image_new)
    # plt.title("Original Image")
    plt.axis("off")
    plt.savefig(
        save_path + "Original Image.png", bbox_inches="tight", pad_inches=0, dpi=300
    )

    # loop over all masks and depth maps
    for nm in Prd:
        if nm in MaskList:
            # copy mask for visualization
            tmIm = Prd[nm][0].copy()
            # save as .npy file
            np.save(save_path + nm + ".npy", tmIm)
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
            plt.imshow(tmIm)
            plt.axis("off")
            plt.savefig(
                save_path + nm + ".png", bbox_inches="tight", pad_inches=0, dpi=300
            )
        else:
            # copy depth map for visualization
            tmIm = Prd[nm].copy()

            # save as .npy file
            np.save(save_path + nm + ".npy", tmIm)

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
            plt.imshow(tmIm, cmap="CMRmap")
            # turn off axis
            plt.axis("off")
            # save image
            plt.savefig(
                save_path + nm + ".png", bbox_inches="tight", pad_inches=0, dpi=300
            )

    print("Results saved in: ", save_path)
