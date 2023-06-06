import numpy as np
import src.models.model as NET_FCN  # The net Class
import torch
import torch.nn as nn
import src.data.make_dataset as MakeDataset

"""
This file contains the evaluation function for the segmentation and depth prediction model

The evaluation function takes in the following arguments:
    --model_path: path to trained model
    --folder_path: path to folder for evaluation
    --UseGPU: True or False (use GPU for evaluation)


"""


def evaluate(
    model_path=r"models/segmentation_depth_model.torch",
    folder_path=r"data/interim/TranProteus1/Testing/LiquidContent",
    UseGPU=True,
):
    TestFolder = folder_path  # input folder

    UseChamfer = False  # Evaluate chamfer distance (this takes lots of time)

    batch_size = 1  # Batch size

    DepthList = [
        "EmptyVessel_Depth",
        "ContentDepth",
        "VesselOpening_Depth",
    ]  # List of depth maps to predict
    MaskList = [
        "VesselMask",
        "ContentMaskClean",
        "VesselOpeningMask",
    ]  # List of segmentation Masks to predict

    # https://arxiv.org/pdf/1406.2283.pdf

    model = NET_FCN.Net(MaskList=MaskList, DepthList=DepthList)
    # Load model weights depending on GPU usage
    if UseGPU == True:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    Readers = MakeDataset.create_reader_Test(batch_size, TestFolder)
    Readers = Readers["Liquid1"]

    MaskEvalType = ["InterSection", "Union", "SumGT", "SumPrd"]
    StatMask = {}
    for nm in MaskList:
        StatMask[nm] = {}
        for et in MaskEvalType:
            StatMask[nm][et] = 0

    # https://arxiv.org/pdf/1406.2283.pdf
    ##https://cseweb.ucsd.edu//~haosu/papers/SI2PC_arxiv_submit.pdf
    EvalTypes = [
        "RMSE",
        "MAE",
        "TSS",
        r"MeanError//GTMaxDist",
        "MeanError//stdv",
        "MeanError//MAD",
        "SumPixels",
    ]  # TSS total sum of squares, RSS Sum of Squares residuals
    if UseChamfer:
        EvalTypes += [
            "ChamferDist//GT_MaxDst",
            "ChamferDist//GT_STDV",
            "ChamferDist//GT_Max_Distance",
        ]
    StatDepth = {}  # Sum All statistics across
    for nm in DepthList:
        StatDepth[nm] = {}
        for et in EvalTypes:
            StatDepth[nm][et] = 0

    while Readers.epoch == 0 and Readers.itr < 1000:  # Test 1000 examples or one epoch
        GT = Readers.LoadSingle()  # Load example

        print(
            "------------------------------",
            Readers.itr,
            "------------------------------",
        )

        with torch.no_grad():
            PrdDepth, PrdProb, PrdMask = model.forward(
                Images=GT["VesselWithContentRGB"]
            )  # Run net inference and get prediction

        for nm in MaskList:
            if nm in GT:
                ROI = GT["ROI"][0]
                Pmask = nn.functional.interpolate(
                    PrdProb[nm],
                    tuple((GT["ROI"].shape[1], GT["ROI"].shape[2])),
                    mode="bilinear",
                    align_corners=False,
                )
                Pmask = (
                    (Pmask[0][1] > 0.5)
                    .squeeze()
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.float32)
                )  # Predicted mask
                Pmask *= ROI  # Limit to the region of interse
                Gmask = GT[nm][0] * ROI  # GT mask  limite to region of interest (ROI)

                # ***************Calculate IOU***********
                InterSection = (Pmask * Gmask).sum()
                Union = (Pmask + Gmask).sum() - InterSection

                if InterSection > 0:
                    StatMask[nm]["Union"] += Union
                    StatMask[nm]["InterSection"] += InterSection
                    StatMask[nm]["SumGT"] += (Gmask).sum()
                    StatMask[nm]["SumPrd"] += (Pmask).sum()

    # ======================Display Segmentation statistics==============================================================================
    print("\n\n\n Segmentation statistics\n")
    for nm in MaskList:
        if StatMask[nm]["Union"] == 0:
            continue
        IOU = StatMask[nm]["InterSection"] / StatMask[nm]["Union"]
        Precision = StatMask[nm]["InterSection"] / (StatMask[nm]["SumPrd"] + 0.0001)
        Recall = StatMask[nm]["InterSection"] / (StatMask[nm]["SumGT"] + 0.0001)
        print(nm, "\tIOU = ", IOU, "\tPrecision = ", Precision, "\tRecall = ", Recall)
