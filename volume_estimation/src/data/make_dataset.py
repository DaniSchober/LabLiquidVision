import sys
import os
import warnings

# add parent folder to sys.path
sys.path.insert(0, os.path.abspath(".."))

# ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ignore RuntimeWarning messages
warnings.filterwarnings("ignore", category=RuntimeWarning)


import segmentation_and_depth.src.models.model as NET_FCN  # The net Class
import torch
import os
import numpy as np
import cv2
import segmentation_and_depth.src.visualization.visualize as vis
import matplotlib.pyplot as plt


# load all folders in data/interim/initial
def create_converted_dataset(
    path_input, path_output, model_path, MaxSize=1000, UseGPU=True
):
    """
    Load all folders in path and return a list of numpy arrays
    """
    # path_input = "data/interim/"
    # path_output = "data/processed/"
    # model_path =  r"../segmentation_and_depth/models/55__03042023-2211.torch" # Trained model path
    # add parent folder to sys.path
    sys.path.insert(0, os.path.abspath(".."))
    # define masks and depth maps to load model
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

    # define masks and depth maps to predict
    DepthList = ["EmptyVessel_Depth", "ContentDepth"]
    MaskList = [
        "VesselMask",
        "ContentMaskClean",
    ]  # List of segmentation Masks to predict
    depth2Mask = {
        "EmptyVessel_Depth": "VesselMask",
        "ContentDepth": "ContentMaskClean",
    }  # Connect depth map to segmentation mask

    count = 0

    to_convert = len(os.listdir(path_input))
    progress = 0

    for folder in os.listdir(path_input):
        # ignore .gitignore
        if folder == ".gitignore":
            continue
        progress += 1
        if progress % 30 == 0:
            print(f"Progress: {progress}/{to_convert}")
        # print(folder)
        folder_path_input = os.path.join(path_input, folder)
        folder_path_output = os.path.join(path_output, folder)
        # create output folder if not exists
        if not os.path.exists(folder_path_output):
            os.makedirs(folder_path_output)
            count += 1
            for file in os.listdir(folder_path_input):
                # join filepath with /
                file_path_input = folder_path_input + "/" + file
                file_path_output = folder_path_output + "/" + file

                # save file in output folder
                if file_path_input.endswith("npy"):
                    np.save(file_path_output, np.load(file_path_input))
                # save file in output folder
                elif file_path_input.endswith("png"):
                    cv2.imwrite(file_path_output, cv2.imread(file_path_input))
                # save file in output folder
                elif file_path_input.endswith("txt"):
                    # open file and save it in output folder
                    with open(file_path_input, "r") as f:
                        with open(file_path_output, "w") as f1:
                            for line in f:
                                f1.write(line)

                if file_path_input.endswith("RGBImage.png"):
                    # Load image
                    image = cv2.imread(file_path_input)
                    image = vis.ResizeToMaxSize(image, MaxSize)
                    image = np.expand_dims(image, axis=0)
                    image_new = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)

                    with torch.no_grad():
                        PrdDepth, PrdProb, PrdMask = model.forward(
                            Images=image, TrainMode=False, UseGPU=UseGPU
                        )

                    Prd = {}
                    for nm in PrdDepth:
                        # convert to numpy and transpose to (H,W,C)
                        Prd[nm] = (
                            (PrdDepth[nm].transpose(1, 2).transpose(2, 3))
                            .data.cpu()
                            .numpy()
                        )
                    for nm in PrdMask:
                        # convert to numpy
                        Prd[nm] = (PrdMask[nm]).data.cpu().numpy()

                    # loop over all masks and depth maps
                    for nm in Prd:
                        if nm in MaskList:
                            np.save(
                                file_path_output.replace("RGBImage.png", nm + ".npy"),
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
                                file_path_output.replace("RGBImage.png", nm + ".png"),
                                image,
                            )

                        elif nm in DepthList:
                            # copy depth map for visualization
                            tmIm = Prd[nm].copy()
                            tmIm = tmIm.squeeze()
                            np.save(
                                file_path_output.replace("RGBImage.png", nm + ".npy"),
                                tmIm,
                            )
                            if nm in depth2Mask:
                                # Remove region out side of the object mask from the depth mask
                                tmIm[Prd[depth2Mask[nm]][0] == 0] = 0
                                np.save(
                                    file_path_output.replace(
                                        "RGBImage.png", nm + "_segmented.npy"
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
                                        file_path_output.replace(
                                            "RGBImage.png", nm + "_segmented.png"
                                        ),
                                        image,
                                    )

                    """
                    Only for visualization and checking of dataset               
                    
                    """
                    Prd = {}
                    for nm in PrdDepth:
                        # convert to numpy and transpose to (H,W,C)
                        Prd[nm] = (
                            (PrdDepth[nm].transpose(1, 2).transpose(2, 3))
                            .data.cpu()
                            .numpy()
                        )
                    for nm in PrdMask:
                        # convert to numpy
                        Prd[nm] = (PrdMask[nm]).data.cpu().numpy()

                    # Visualize results
                    count_vis = 2
                    # create subplot to show all images in one window
                    plt.figure(figsize=(15, 10))
                    # show original image
                    plt.subplot(3, 2, 1)
                    plt.imshow(image_new)
                    plt.title("Original Image")
                    plt.axis("off")
                    # loop over all masks and depth maps
                    for nm in Prd:
                        if nm in MaskList:
                            count_vis += 1
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
                            plt.subplot(3, 2, count_vis)
                            plt.imshow(tmIm)
                            plt.axis("off")
                            plt.title(nm)
                        elif nm in DepthList:
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
                            plt.subplot(3, 2, count_vis)
                            plt.imshow(tmIm, cmap="CMRmap")
                            # turn off axis
                            plt.axis("off")
                            plt.title(nm)
                    plt.savefig(
                        file_path_output.replace("RGBImage.png", "visualize.png")
                    )

    print("Created {} new folders.".format(count))
    print("Total size of dataset: {} folders.".format(len(os.listdir(path_output))))
