import numpy as np
import src.models.model as NET_FCN  # The net Class
import torch
import torch.nn as nn
import src.data.make_dataset as MakeDataset
from src.utils import metrics

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
    depth2Mask = {
        "EmptyVessel_Depth": "VesselMask",
        "ContentDepth": "ContentMaskClean",
        "VesselOpening_Depth": "VesselOpeningMask",
    }

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
        "abs_rel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "log10",
        "silog",
    ]  # List of evaluation metrics

    StatDepth = {}  # Sum All statistics across
    for nm in DepthList:
        StatDepth[nm] = {}
        for et in EvalTypes:
            StatDepth[nm][et] = 0

    # create an array of length DepthList to store the counts
    count = np.zeros(len(DepthList))

    StatDepth = {}  # Sum All statistics across
    for nm in DepthList:
        StatDepth[nm] = {}
        for et in EvalTypes:
            StatDepth[nm][et] = 0

    i = 0

    while Readers.epoch == 0 and Readers.itr < 500:  # Test 1000 examples or one epoch
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

        ########################################### Evaluate Mask #######################################################################
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

                InterSection = (Pmask * Gmask).sum()
                Union = (Pmask + Gmask).sum() - InterSection

                if InterSection > 0:
                    StatMask[nm]["Union"] += Union
                    StatMask[nm]["InterSection"] += InterSection
                    StatMask[nm]["SumGT"] += (Gmask).sum()
                    StatMask[nm]["SumPrd"] += (Pmask).sum()

        i = 0

        # ########################################## Evaluate Depth #######################################################################
        for nm in DepthList:
            if nm in GT:
                ROI = GT["ROI"][0]

                # get segmentation mask
                Gmask = (
                    GT[depth2Mask[nm]][0] * ROI
                )  # GT mask  limite to region of ineterstr (ROI)

                # get prediction of depth map
                Pdepth = nn.functional.interpolate(
                    PrdDepth[nm],
                    tuple((GT["ROI"].shape[1], GT["ROI"].shape[2])),
                    mode="bilinear",
                    align_corners=False,
                )
                Pdepth = Pdepth[0][0].cpu().detach().numpy().astype(np.float32)
                Pdepth *= ROI  # Limit to the region of intersection
                # get only the region of segmentation mask
                Pdepth = Pdepth * Gmask

                Gdepth = (
                    GT[nm][0] * ROI
                )  # GT depth  limite to region of intersection (ROI)
                # get only the region of segmentation mask

                Gdepth = Gdepth * Gmask
                scale_factor = Gdepth[Gdepth > 0].mean() / Pdepth[Pdepth > 0].mean()
                Pdepth *= scale_factor

                # convert to torch tensor
                Pdepth = torch.from_numpy(Pdepth)
                Gdepth = torch.from_numpy(Gdepth)

                ROI = torch.autograd.Variable(
                    torch.from_numpy(GT[depth2Mask[nm]] * GT["ROI"]).unsqueeze(1),
                    requires_grad=False,
                )  # ROI to torch
                ROI[
                    ROI < 0.9
                ] = 0  # Resize have led to some intirmidiate values ignore them
                ROI[
                    ROI > 0.9
                ] = 1  # Resize have led to some intirmidiate values ignore them

                
                # print(dic)

                if Pdepth[Pdepth != 0] is not None and Gdepth[Gdepth != 0] is not None and  Pdepth[Pdepth != 0].shape ==  Gdepth[Gdepth != 0].shape:
                    # calculate metrics
                    dic = metrics.eval_depth(Pdepth, Gdepth)
                    # add results to dictionary
                    for et in EvalTypes:
                        StatDepth[nm][et] += dic[et]
                    # add one to the count
                    count[i] += 1
                    i += 1

    # ======================Display Segmentation statistics==============================================================================
    print("\n\n\n Segmentation statistics\n")

    for nm in MaskList:
        if StatMask[nm]["Union"] == 0:
            continue
        IOU = StatMask[nm]["InterSection"] / StatMask[nm]["Union"]
        Precision = StatMask[nm]["InterSection"] / (StatMask[nm]["SumPrd"] + 0.0001)
        Recall = StatMask[nm]["InterSection"] / (StatMask[nm]["SumGT"] + 0.0001)
        print(nm, "\tIOU = ", IOU, "\tPrecision = ", Precision, "\tRecall = ", Recall)

    # get means of the dictionary
    i = 0
    print("Count", count)
    print("Mean results")
    for nm in DepthList:
        for et in EvalTypes:
            # print("Stat", StatDepth[nm][et])

            StatDepth[nm][et] /= count[i]
            # print("count", count[i])
            print(nm, et, StatDepth[nm][et])
        i += 1

    # save results to file
    with open("output/results.txt", "w") as f:
        print("Count", count[0], file=f)
        print("--------------------", file=f)
        print("Depth Estimation results:", file=f)
        for nm in DepthList:
            for et in EvalTypes:
                print(nm, et, StatDepth[nm][et], file=f)
            i += 1

        print("--------------------", file=f)
        print("Segmentation results:", file=f)
        for nm in MaskList:
            if StatMask[nm]["Union"] == 0:
                continue
            IOU = StatMask[nm]["InterSection"] / StatMask[nm]["Union"]
            Precision = StatMask[nm]["InterSection"] / (StatMask[nm]["SumPrd"] + 0.0001)
            Recall = StatMask[nm]["InterSection"] / (StatMask[nm]["SumGT"] + 0.0001)
            print(
                nm,
                "\tIOU = ",
                IOU,
                "\tPrecision = ",
                Precision,
                "\tRecall = ",
                Recall,
                file=f,
            )
