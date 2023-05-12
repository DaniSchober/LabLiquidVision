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
from tqdm import tqdm


def train(batch_size, num_epochs, load_pretrained_model, use_labpics):
    Trained_model_path = ""  # Path of trained model weights if you want to return to trained model, else if there is no pretrained mode this should be =""
    Learning_Rate = 1e-5  # intial learning rate
    TrainedModelWeightDir = "logs1/"  # Output Folder where trained model weight and information will be stored

    TrainLossTxtFile = (
        TrainedModelWeightDir + "TrainLoss_" + time.strftime("%d%m%Y-%H%M") + ".txt"
    )  # Where train losses statistics will be written

    Weight_Decay = 4e-5  # Weight for the weight decay loss function

    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )  # Use GPU if available

    # device = "cpu"
    print("Device used: ", device)

    # List of depth maps and segmentation Mask to predict
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
    }  # Dictionary connecting depth maps and segmentation maps of same object

    # Create and initiate net
    Net = NET_FCN.Net(
        MaskList=MaskList, DepthList=DepthList
    )  # Create net and load pretrained

    # Load net weights of previous training if exist
    InitStep = 1
    InitEpoch = 1
    if load_pretrained_model:
        if os.path.exists("models/40__29032023-0231.torch"):
            Trained_model_path = "models/40__29032023-0231.torch"
            print("Loading pretrained model...")
            print("Trained_model_path: ", Trained_model_path)
        if os.path.exists(TrainedModelWeightDir + "/Learning_Rate.npy"):
            Learning_Rate = np.load(TrainedModelWeightDir + "/Learning_Rate.npy")
        if os.path.exists(TrainedModelWeightDir + "/epoch.npy"):
            InitEpoch = int(np.load(TrainedModelWeightDir + "/epoch.npy"))
    if (
        Trained_model_path != ""
    ):  # Optional initiate full net by loading a pretrained net
        Net.load_state_dict(torch.load(Trained_model_path))

    Net = Net.to(device)  # Send net to GPU if available
    # print net details
    print("Net: ", Net)
    # print gpu usage
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=Net.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay
    )

    # Create reader for data sets
    Readers = MakeDataset.create_reader(batch_size)
    if use_labpics:
        LPReaders = MakeDataset.create_reader_LabPics(batch_size)

    # get number of samples
    num_samples = MakeDataset.get_num_samples(
        Readers
    ) + MakeDataset.get_num_samples_LabPics(LPReaders)
    print("Num_samples ", num_samples)

    itr_per_epoch = int(num_samples / batch_size)

    # Create logs files for saving loss during training

    if not os.path.exists(TrainedModelWeightDir):
        os.makedirs(TrainedModelWeightDir)  # Create folder for trained weight
    torch.save(
        Net.state_dict(), TrainedModelWeightDir + "/" + "test" + ".torch"
    )  # test saving to see that everything is fine

    # Loss Parameters
    AVGCatLoss = {}  # Average loss for each prediction

    # Start training loop
    print("Start Training")

    for epoch_num in range(InitEpoch, num_epochs):
        print("Epoch ", epoch_num)
        for itr in tqdm(range(1, itr_per_epoch)):  # tqdm for progress bar
            # read batch
            Mode = "Virtual"  # Transproteus data

            if use_labpics and np.random.rand() < 0.33:
                Mode = "LabPics"  # randomly selecting dataset

            if Mode == "Virtual":  # Read transproteus
                readertype = list(Readers)[
                    np.random.randint(len(list(Readers)))
                ]  # Pick reader (folder)
                GT = Readers[readertype].LoadBatch()

            if Mode == "LabPics":  # Read Labpics data
                readertype = list(LPReaders)[
                    np.random.randint(len(list(LPReaders)))
                ]  # Pick reader (folder)
                GT = LPReaders[readertype].LoadBatch()

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

            # Calculate loss

            CatLoss = {}  # will store the Category loss per object

            # Depth map loss

            ###############
            # Dani: changed this to depth loss!
            ###############

            if Mode == "Virtual":  # depth loss is calculated only for transproteus
                TGT = {}  # GT depth in torch format
                for nm in DepthList:
                    # ROI Punish depth prediction only within the object mask, resize ROI to prediction size (prediction map is shrink version of the input image)
                    ROI = torch.autograd.Variable(
                        torch.from_numpy(GT[depth2Mask[nm]] * GT["ROI"])
                        .unsqueeze(1)
                        .to(device),
                        requires_grad=False,
                    )  # ROI to torch
                    ROI = nn.functional.interpolate(
                        ROI,
                        tuple(
                            (
                                PrdDepth[nm].shape[2],
                                PrdDepth["EmptyVessel_Depth"].shape[3],
                            )
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
                    )  # GT Depth log
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

            # Segmentation mask loss
            ROI = torch.autograd.Variable(
                torch.from_numpy(GT["ROI"]).unsqueeze(1).to(device), requires_grad=False
            )  # Region of interest in the image where loss is calulated
            ROI = nn.functional.interpolate(
                ROI,
                tuple((PrdProb[MaskList[0]].shape[2], PrdProb[MaskList[0]].shape[3])),
                mode="bilinear",
                align_corners=False,
            )  # Resize ROI to prediction size (prediction map is shrink version of the input image)

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

            fr = 1 / np.min(
                [itr + (epoch_num - 1) * itr_per_epoch - InitStep + 1, 2000]
            )
            TotalLoss = 0  # will be used for backpropagation
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
                    AVGCatLoss[nm] = CatLoss[nm].data.cpu().numpy()
                TotalLoss += CatLoss[nm]

                if "Depth" in nm:
                    AVGCatLoss["Depth"] += AVGCatLoss[nm]
                if "Mask" in nm:
                    AVGCatLoss["Mask"] += AVGCatLoss[nm]
                AVGCatLoss["Total"] += AVGCatLoss[nm]

            # Apply backpropogation
            TotalLoss.backward()  # Backpropogate loss
            optimizer.step()  # Apply gradient descent change to weight

            # print results to file every 4 iterations
            if itr % 4 == 0 and Mode == "Virtual":
                txt = "\n" + str((epoch_num - 1) * itr_per_epoch + itr)
                for nm in AVGCatLoss:
                    txt += (
                        "\tAverage Cat Loss ["
                        + nm
                        + "] "
                        + str(float("{:.4f}".format(AVGCatLoss[nm])))
                    )
                # print(txt)
                # Write train loss to file
                with open(TrainLossTxtFile, "a") as f:
                    f.write(txt)
                    f.close()

        # Save trained model after each epoch
        print("Saving Model to file in " + TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBackUp.torch")
        print("Model saved.")
        np.save(TrainedModelWeightDir + "/Learning_Rate.npy", Learning_Rate)
        np.save(TrainedModelWeightDir + "/epoch.npy", epoch_num)
        if (
            epoch_num % 5 == 0
        ):  # Save model weight once every 5 epochs (to save space and time) and at the end of training
            print(
                "Saving Model to file in " + "models" + "/" + str(epoch_num) + ".torch"
            )
            torch.save(
                Net.state_dict(),
                "models"
                + "/"
                + str(epoch_num)
                + "__"
                + time.strftime("%d%m%Y-%H%M")
                + ".torch",
            )
            print("model saved")

        # Update learning rate
        if epoch_num % 5 == 0:
            if "TotalPrevious" not in AVGCatLoss:
                AVGCatLoss["TotalPrevious"] = AVGCatLoss["Total"]
            elif (
                AVGCatLoss["Total"] * 0.95 < AVGCatLoss["TotalPrevious"]
            ):  # If average loss did not decrease in the last 10 epochs update training loss
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
        tqdm.write(f"Epoch {epoch_num} completed")
