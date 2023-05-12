# Train net that predict depth map and segmentation of vessel, vessel content, and vessel opening
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.models.model as NET_FCN  # The net Class
import src.data.load_data as DepthReader
import src.data.make_dataset as MakeDataset
import src.models.loss_functions as LossFunctions
import time


def train():
    Trained_model_path = ""  # Path of trained model weights if you want to return to trained model, else if there is no pretrained mode this should be =""
    Learning_Rate = 1e-5  # intial learning rate
    TrainedModelWeightDir = "logs/"  # Output Folder where trained model weight and information will be stored

    TrainLossTxtFile = (
        TrainedModelWeightDir + "TrainLoss_" + time.strftime("%d%m%Y-%H%M") + ".txt"
    )  # Where train losses statistics will be written

    Weight_Decay = 4e-5  # Weight for the weight decay loss function
    MAX_ITERATION = int(100000010)  # Max number of training iteration

    # device = "cpu"
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )  # Use GPU if available
    print("Device", device)

    ################## List of depth maps and segmentation Mask to predict ###################################################################################################################3

    MaskClasses = {}
    # XYZList = ["VesselXYZ","ContentXYZ","VesselOpening_XYZ"] # List of XYZ maps to predict
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
    # XYZ2Mask={"VesselXYZ":"VesselMask","ContentXYZ":"ContentMaskClean","VesselOpening_XYZ":"VesselOpeningMask"} # Dictionary connecting XYZ maps and segmentation maps of same object
    depth2Mask = {
        "EmptyVessel_Depth": "VesselMask",
        "ContentDepth": "ContentMaskClean",
        "VesselOpening_Depth": "VesselOpeningMask",
    }  # Dictionary connecting depth maps and segmentation maps of same object
    # XYZ2LossFactor={"VesselXYZ":1,"ContentXYZ":1.,"VesselOpening_XYZ":0.4} # Weight of loss of XYZ prediction per object (some object will contribute less to the loss function)
    depth2LossFactor = {
        "EmptyVessel_Depth": 1,
        "ContentDepth": 1.0,
        "VesselOpening_Depth": 0.4,
    }  # Weight of loss of depth prediction per object (some object will contribute less to the loss function)

    # =========================Load net weights====================================================================================================================
    InitStep = 1
    if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
        Trained_model_path = TrainedModelWeightDir + "/Defult.torch"
    if os.path.exists(TrainedModelWeightDir + "/Learning_Rate.npy"):
        Learning_Rate = np.load(TrainedModelWeightDir + "/Learning_Rate.npy")
    if os.path.exists(TrainedModelWeightDir + "/itr.npy"):
        InitStep = int(np.load(TrainedModelWeightDir + "/itr.npy"))

    #################### Create and Initiate net and create optimizer ##########################################################################################3

    Net = NET_FCN.Net(
        MaskList=MaskList, DepthList=DepthList
    )  # Create net and load pretrained

    # --------------------if previous model exist load it--------------------------------------------------------------------------------------------
    if (
        Trained_model_path != ""
    ):  # Optional initiate full net by loading a pretrained net
        Net.load_state_dict(torch.load(Trained_model_path))
    Net = Net.to(device)  # Send net to GPU if available
    # --------------------------------Optimizer--------------------------------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        params=Net.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay
    )  # Create adam optimizer

    # ----------------------------------------Create reader for data sets--------------------------------------------------------------------------------------------------------------
    """
    Readers = {}  # Transproteus readers
    for nm in TransProteusFolder:
        Readers[nm] = DepthReader.Reader(
            TransProteusFolder[nm],
            MaxBatchSize,
            MinSize,
            MaxSize,
            MaxPixels,
            TrainingMode=True,
        )

    """

    Readers = MakeDataset.create_reader()  # Create readers for datasets

    # --------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------

    if not os.path.exists(TrainedModelWeightDir):
        os.makedirs(TrainedModelWeightDir)  # Create folder for trained weight
    torch.save(
        Net.state_dict(), TrainedModelWeightDir + "/" + "test" + ".torch"
    )  # test saving to see the everything is fine

    # f = open(TrainLossTxtFile, "w+")  # Training loss log file
    # f.write("Iteration\tloss\t Learning Rate=")
    # f.close()
    # -------------------Loss Parameters--------------------------------------------------------------------------------
    PrevAvgLoss = (
        0  # Average loss in the past (to compare see if loss as been decrease)
    )
    AVGCatLoss = {}  # Average loss for each prediction

    ############################################################################################################################
    # ..............Start Training loop: Main Training....................................................................
    print("Start Training")
    for itr in range(InitStep, MAX_ITERATION):  # Main training loop
        print(
            "------------------------------ Iteration: ",
            itr,
            "------------------------------------------------",
        )

        # ***************************Reading batch ******************************************************************************
        Mode = "Virtual"  # Transproteus data
        if Mode == "Virtual":  # Read transproteus
            readertype = list(Readers)[
                np.random.randint(len(list(Readers)))
            ]  # Pick reader (folder)
            # print(readertype)
            GT = Readers[readertype].LoadBatch()  # Read batch

        print("Run prediction")

        # PrdXYZ, PrdProb, PrdMask = Net.forward(Images=GT["VesselWithContentRGB"]) # Run net inference and get prediction
        PrdDepth, PrdProb, PrdMask = Net.forward(
            Images=GT["VesselWithContentRGB"]
        )  # Run net inference and get prediction

        Net.zero_grad()

        for nm in MaskList:
            # change to device
            PrdMask[nm] = PrdMask[nm].to(device)
            PrdProb[nm] = PrdProb[nm].to(device)

        for nm in DepthList:
            PrdDepth[nm] = PrdDepth[nm].to(device)

        # ------------------------Calculating loss---------------------------------------------------------------------

        CatLoss = {}  # will store the Category loss per object

        # **************************************Depth Map Loss*************************************************************************************************************************

        ###############
        # Dani: change this to depth loss!!!!!!
        ###############
        ###############

        if Mode == "Virtual":  # depth loss is calculated only for transproteus
            print("Calculating Depth Loss")
            TGT = {}  # GT depth in torch format

            NormConst = []  # Scale constant to normalize the predicted depth map
            for nm in DepthList:
                # ------------------------ROI Punish depth prediction only within  the object mask, resize  ROI to prediction size (prediction map is shrink version of the input image)----------------------------------------------------
                ROI = torch.autograd.Variable(
                    torch.from_numpy(GT[depth2Mask[nm]] * GT["ROI"])
                    .unsqueeze(1)
                    .to(device),
                    requires_grad=False,
                )  # ROI to torch
                ROI = nn.functional.interpolate(
                    ROI,
                    tuple(
                        (PrdDepth[nm].shape[2], PrdDepth["EmptyVessel_Depth"].shape[3])
                    ),
                    mode="bilinear",
                    align_corners=False,
                )  # ROI to output scale
                ROI[
                    ROI < 0.9
                ] = 0  # Resize have led to some intirmidiate values ignore them
                ROI[
                    ROI > 0.9
                ] = 1  # Resize have led to some intirmidiate values ignore them

                TGT[nm] = torch.log(
                    torch.from_numpy(GT[nm]).to(device).unsqueeze(1) + 0.0001
                )  ### GT Depth log
                TGT[nm].requires_grad = False
                TGT[nm] = nn.functional.interpolate(
                    TGT[nm],
                    tuple((PrdDepth[nm].shape[2], PrdDepth[nm].shape[3])),
                    mode="bilinear",
                    align_corners=False,
                )  # convert to prediction size

                CatLoss[nm] = 5 * LossFunctions.DepthLoss(
                    PrdDepth[nm], TGT[nm], ROI
                )  # Loss function

        ###############################################################################################################################################
        # ******************Segmentation Mask Loss************************************************************************************************************************************
        # -----------------------------ROI---------------------------------------------------------------------------
        ROI = torch.autograd.Variable(
            torch.from_numpy(GT["ROI"]).unsqueeze(1).to(device), requires_grad=False
        )  # Region of interest in the image where loss is calulated
        ROI = nn.functional.interpolate(
            ROI,
            tuple((PrdProb[MaskList[0]].shape[2], PrdProb[MaskList[0]].shape[3])),
            mode="bilinear",
            align_corners=False,
        )  # Resize ROI to prediction
        # -------------------
        print("Calculating Mask Loss")

        for nm in MaskList:
            if nm in GT:
                TGT = torch.autograd.Variable(
                    torch.from_numpy(GT[nm]).to(device), requires_grad=False
                ).unsqueeze(
                    1
                )  # Convert GT segmentation mask to pytorch
                TGT = nn.functional.interpolate(
                    TGT,
                    tuple((PrdProb[nm].shape[2], PrdProb[nm].shape[3])),
                    mode="bilinear",
                    align_corners=False,
                )  # Resize GT mask to predicted image size (prediction is scaled down version of the image)
                CatLoss[nm] = -torch.mean(
                    TGT[:, 0] * torch.log(PrdProb[nm][:, 1] + 0.00001) * ROI[:, 0]
                ) - torch.mean(
                    (1 - TGT[:, 0])
                    * torch.log(PrdProb[nm][:, 0] + 0.0000001)
                    * ROI[:, 0]
                )  # Calculate cross entropy loss

        # ==========================================================================================================================
        # ---------------Calculate Total Loss and average loss by using the sum of all objects losses----------------------------------------------------------------------------------------------------------
        print("Calculating Total Loss")
        fr = 1 / np.min([itr - InitStep + 1, 2000])
        TotalLoss = 0  # will be used for backptop
        AVGCatLoss["Depth"] = 0  # will be used to collect statitics
        AVGCatLoss["Mask"] = 0  # will be used to collect statitics
        AVGCatLoss["Total"] = 0  # will be used to collect statitics
        for nm in CatLoss:  # Go over all object losses and sum them
            if not nm in AVGCatLoss:
                AVGCatLoss[nm] = 0
            if CatLoss[nm] > 0:
                AVGCatLoss[nm] = (1 - fr) * AVGCatLoss[nm] + fr * CatLoss[
                    nm
                ].data.cpu().numpy()
            TotalLoss += CatLoss[nm]

            if "Depth" in nm:
                AVGCatLoss["Depth"] += AVGCatLoss[nm]
            if "Mask" in nm:
                AVGCatLoss["Mask"] += AVGCatLoss[nm]
            AVGCatLoss["Total"] += AVGCatLoss[nm]
        # --------------Apply backpropogation-----------------------------------------------------------------------------------
        print("Back Propagation")
        TotalLoss.backward()  # Backpropogate loss
        optimizer.step()  # Apply gradient descent change to weight
        print("Done")

        ###############################################################################################################################
        # ===================Display, Save and update learning rate======================================================================================
        #########################################################################################################################33

        # --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
        if (
            itr % 3000 == 0
        ):  # and itr>0: #Save model weight once every 300 steps, temp file
            print("Saving Model to file in " + TrainedModelWeightDir + "/Defult.torch")
            torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
            torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
            print("model saved")
            np.save(TrainedModelWeightDir + "/Learning_Rate.npy", Learning_Rate)
            np.save(TrainedModelWeightDir + "/itr.npy", itr)
        if (
            itr % 60000 == 0 and itr > 0
        ):  # Save model weight once every 60k steps permenant file
            print(
                "Saving Model to file in "
                + TrainedModelWeightDir
                + "/"
                + str(itr)
                + ".torch"
            )
            torch.save(
                Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch"
            )
            print("model saved")
        # ......................Write and display train loss..........................................................................
        if itr % 2 == 0:  # Display train loss and write to statics file
            txt = "\n" + str(itr)
            for nm in AVGCatLoss:
                txt += (
                    "\tAverage Cat Loss ["
                    + nm
                    + "] "
                    + str(float("{:.4f}".format(AVGCatLoss[nm])))
                    + "  "
                )
                # get two decimal places of AVGCatLoss
                # AVGCatLoss[nm] = float("{:.2f}".format(AVGCatLoss[nm]))
            if itr % 10 == 0:
                print(txt)
            # Write train loss to file
            with open(TrainLossTxtFile, "a") as f:
                f.write(txt)
                f.close()
        # #----------------Update learning rate -------------------------------------------------------------------------------
        if itr % 20000 == 0:
            if "TotalPrevious" not in AVGCatLoss:
                AVGCatLoss["TotalPrevious"] = AVGCatLoss["Total"]
            elif (
                AVGCatLoss["Total"] * 0.95 < AVGCatLoss["TotalPrevious"]
            ):  # If average loss did not decrease in the last 20k steps update training loss
                Learning_Rate *= 0.9  # Reduce learning rate
                if Learning_Rate <= 3e-7:  # If learning rate to small increae it
                    Learning_Rate = 5e-6
                print("Learning Rate=" + str(Learning_Rate))
                print(
                    "======================================================================================================================"
                )
                optimizer = torch.optim.Adam(
                    params=Net.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay
                )  # Create adam optimizer with new learning rate
                if device == "cuda":
                    torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
            AVGCatLoss["TotalPrevious"] = (
                AVGCatLoss["Total"] + 0.0000000001
            )  # Save current average loss for later comparison
