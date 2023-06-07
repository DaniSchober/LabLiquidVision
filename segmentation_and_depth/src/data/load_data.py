import json
import os
import threading
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

'''
This file contains the data reader classes for the segmentation and depth estimation project

The reader reads the data from the dataset and prepares it for training and testing


'''


MapsAndDepths = {  # List of maps to use and their depths (layers)
    "VesselMask": 1,
    "VesselOpening_Depth": 1,
    "VesselWithContentRGB": 3,
    "VesselWithContentNormal": 3,
    "VesselWithContentDepth": 1,
    "EmptyVessel_Depth": 1,
    "ContentNormal": 3,
    "ContentDepth": 1,
    "ContentMask": 3,
    "ContentMaskClean": 1,
    "VesselOpeningMask": 1,
    "ROI": 1,
}


class Reader:
    """
    This class is used to read the data from the TransProteus dataset

    MainDir: The main directory of the dataset
    MaxBatchSize: The maximum number of images in the batch
    MinSize: The minimum size of the image in pixels
    MaxSize: The maximum size of the image in pixels
    MaxPixels: The maximum number of pixels in the batch
    TrainingMode: If true the reader will read the data in training mode (randomly) if false it will read the data in testing mode (sequentially)

    """

    def __init__(
        self,
        MainDir=r"",
        MaxBatchSize=1,
        MinSize=250,
        MaxSize=900,
        MaxPixels=800 * 800 * 5,
        TrainingMode=True,
    ):
        self.MaxBatchSize = MaxBatchSize  # Max number of image in batch
        self.MinSize = MinSize  # Min image width and hight in pixels
        self.MaxSize = MaxSize  # Max image width and hight in pixels
        self.MaxPixels = MaxPixels  # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.epoch = 0  # Training Epoch
        self.itr = 0  # Training iteratation
        # Create list of annotations with maps and dicitionaries per annotation
        self.AnnList = []  # Image/annotation list
        print("Creating annotation list for reader. This might take a while.")
        for AnnDir in os.listdir(MainDir):
            if AnnDir != ".DS_Store":
                AnnDir = MainDir + "//" + AnnDir + "//"
                Ent = {}
                if os.path.isfile(AnnDir + "//ContentMaterial.json"):
                    Ent["ContentMaterial"] = AnnDir + "//ContentMaterial.json"
                if os.path.isfile(AnnDir + "//VesselMaterial.json"):
                    Ent["VesselMaterial"] = AnnDir + "//VesselMaterial.json"
                if os.path.isfile(AnnDir + "//CameraParameters.json"):
                    Ent["CameraParameters"] = AnnDir + "//CameraParameters.json"
                Ent["VesselMask"] = AnnDir + "//VesselMask.png"
                Ent["VesselOpening_Depth"] = AnnDir + "//VesselOpening_Depth.exr"
                Ent["EmptyVessel_RGB"] = AnnDir + "//EmptyVessel_Frame_0_RGB.jpg"
                Ent["EmptyVessel_Normal"] = AnnDir + "//EmptyVessel_Frame_0_Normal.exr"
                Ent["EmptyVessel_Depth"] = AnnDir + "//EmptyVessel_Frame_0_Depth.exr"
                Ent["MainDir"] = AnnDir
                for nm in os.listdir(AnnDir):
                    filepath = AnnDir + "/" + nm
                    if ("VesselWithContent" in nm) and ("_RGB.jpg" in nm):
                        EntTemp = Ent.copy()
                        EntTemp["VesselWithContentRGB"] = filepath
                        EntTemp["VesselWithContentNormal"] = filepath.replace(
                            "_RGB.jpg", "_Normal.exr"
                        )
                        EntTemp["VesselWithContentDepth"] = filepath.replace(
                            "_RGB.jpg", "_Depth.exr"
                        )
                        EntTemp["ContentRGB"] = EntTemp["VesselWithContentRGB"].replace(
                            "VesselWithContent_", "Content_"
                        )
                        EntTemp["ContentNormal"] = EntTemp[
                            "VesselWithContentNormal"
                        ].replace("VesselWithContent_", "Content_")
                        EntTemp["ContentDepth"] = EntTemp[
                            "VesselWithContentDepth"
                        ].replace("VesselWithContent_", "Content_")
                        EntTemp["ContentMask"] = EntTemp["ContentDepth"].replace(
                            "_Depth.exr", "_Mask.png"
                        )
                        self.AnnList.append(EntTemp)
        # Check list for errors
        for Ent in self.AnnList:
            for nm in Ent:
                # print(Ent[nm])
                if (".exr" in Ent[nm]) or (".png" in Ent[nm]) or (".jpg" in Ent[nm]):
                    if os.path.exists(Ent[nm]):
                        pass
                    else:
                        exit()

        print(
            "Done making file list.\nTotal number of samples = "
            + str(len(self.AnnList))
        )
        if TrainingMode:
            self.StartLoadBatch()
        self.AnnData = False

    def GetNumSamples(self):
        '''
        This function returns the number of samples in the dataset
        '''
        return len(self.AnnList)

    # Crop and resize image and mask and ROI to fit batch size
    def CropResize(self, Maps, Hb, Wb):
        '''
        This function crops and resizes the image and mask and ROI to fit the batch size

        Input:
            --Maps: dictionary of maps
            --Hb: height of batch
            --Wb: width of batch
        
        Output:
            --Maps: dictionary of maps

        '''
        # resize image if it too small to the batch size

        h, w = Maps["ROI"].shape
        Bs = np.min((h / Hb, w / Wb))
        if (
            Bs < 1 or Bs > 3 or np.random.rand() < 0.2
        ):  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(h / Bs) + 1
            w = int(w / Bs) + 1
            for nm in Maps:
                if hasattr(Maps[nm], "shape"):  # check if array
                    if "RGB" in nm:
                        Maps[nm] = cv2.resize(
                            Maps[nm], dsize=(w, h), interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        Maps[nm] = cv2.resize(
                            Maps[nm], dsize=(w, h), interpolation=cv2.INTER_NEAREST
                        )
        # Crop image to fit batch size

        if w > Wb:
            X0 = np.random.randint(w - Wb)  # int((w - Wb)/2-0.1)#
        else:
            X0 = 0
        if h > Hb:
            Y0 = np.random.randint(h - Hb)  # int((h - Hb)/2-0.1)#
        else:
            Y0 = 0

        for nm in Maps:
            if hasattr(Maps[nm], "shape"):  # check if array
                Maps[nm] = Maps[nm][Y0 : Y0 + Hb, X0 : X0 + Wb]

        # If still not batch size resize again
        for nm in Maps:
            if hasattr(Maps[nm], "shape"):  # check if array
                if not (Maps[nm].shape[0] == Hb and Maps[nm].shape[1] == Wb):
                    Maps[nm] = cv2.resize(
                        Maps[nm], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST
                    )
        return Maps

    def Augment(self, Maps):
        '''
        This function augments the image and mask

        It applies the following augmentations for 10% of the images:
            --Gaussian blur
            --Dark light
            --GreyScale

        Input:
            --Maps: dictionary of maps

        Output:
            --Maps: dictionary of augmented maps

        '''
        for nm in Maps:
            if "RGB" in nm:
                if np.random.rand() < 0.1:  # Gaussian blur (10% of the time)
                    Maps[nm] = cv2.GaussianBlur(Maps[nm], (5, 5), 0)

                if np.random.rand() < 0.1:  # Dark light (10% of the time)
                    Maps[nm] = Maps[nm] * (0.5 + np.random.rand() * 0.65)
                    Maps[nm][Maps[nm] > 255] = 255

                if np.random.rand() < 0.1:  # GreyScale (10% of the time)
                    Gr = Maps[nm].mean(axis=2)
                    r = np.random.rand()

                    Maps[nm][:, :, 0] = Maps[nm][:, :, 0] * r + Gr * (1 - r)
                    Maps[nm][:, :, 1] = Maps[nm][:, :, 1] * r + Gr * (1 - r)
                    Maps[nm][:, :, 2] = Maps[nm][:, :, 2] * r + Gr * (1 - r)

        return Maps

    def LoadNext(self, pos, Hb, Wb):

        '''
        This function reads the next image annotation and data

        Input:
            --pos: position in batch
            --Hb: height of batch
            --Wb: width of batch

        Output:
            --Maps: dictionary of maps

        '''
        # Select random example from the batch
        AnnInd = np.random.randint(len(self.AnnList))
        Ann = self.AnnList[AnnInd]
        Maps = {}  # List of all maps to read

        with open(Ann["CameraParameters"]) as f:
            Maps["CameraParameters"] = json.load(f)

        for nm in MapsAndDepths:  # Load segmentation maps and depth maps
            if not nm in Ann:
                continue
            Path = Ann[nm]
            Depth = MapsAndDepths[nm]
            if ".exr" in Path:  # Depth maps and normal maps
                I = cv2.imread(Path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if np.ndim(I) >= 3 and Depth == 1:
                    I = I[:, :, 0]
            else:  # Segmentation mask
                if Depth == 1:
                    I = cv2.imread(Path, 0)
                else:
                    I = cv2.imread(Path)
            Maps[nm] = I.astype(np.float32)

        # Process segmentation mask
        Maps["VesselMask"][Maps["VesselMask"] > 0] = 1
        Maps["VesselOpeningMask"] = (Maps["VesselOpening_Depth"] < 5000).astype(
            np.float32
        )
        Maps["ContentMaskClean"] = (Maps["ContentMask"].sum(2) > 0).astype(np.float32)
        Maps["ROI"] = np.ones(Maps["VesselMask"].shape, dtype=np.float32)

        IgnoreMask = Maps["ContentMask"][
            :, :, 2
        ]  # Undistorted content not viewed trough the vessel walls is ignored (leaking)
        IgnoreMask[
            Maps["ContentMask"][:, :, 1] > 0
        ] = 0  # Contet viewed trough vessel opening is not ignored
        IgnoreMask[
            (Maps["ContentMask"][:, :, 1] * Maps["ContentMask"][:, :, 0]) > 0
        ] = 1  # areas where the content is viewd trough the vessel floor are ignored
        Maps["ROI"][
            IgnoreMask > 0
        ] = 0  # Region of interest where the annotation is well defined
        # Generate depth map
        Maps["EmptyVessel_Depth"][
            Maps["EmptyVessel_Depth"] > 5000
        ] = 0  # Remove far away background points
        Maps["VesselOpening_Depth"][
            Maps["VesselOpening_Depth"] > 5000
        ] = 0  # Remove far away background points
        Maps["ContentDepth"][
            Maps["ContentDepth"] > 5000
        ] = 0  # Remove far away background points

        Maps = self.Augment(Maps)
        if Hb != -1:
            Maps = self.CropResize(Maps, Hb, Wb)

        for nm in Maps:
            if nm in self.Maps:
                self.Maps[nm][pos] = Maps[nm]

    # Start load batch of images
    def StartLoadBatch(self):

        '''
        This function starts loading the next batch of images

        '''
        # Initiate batch
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)
            if Hb * Wb < self.MaxPixels:
                break
        BatchSize = np.intc(
            np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize))
        )
        # Create empty maps batch
        self.Maps = {}
        for nm in MapsAndDepths:
            if MapsAndDepths[nm] > 1:
                self.Maps[nm] = np.zeros(
                    [BatchSize, Hb, Wb, MapsAndDepths[nm]], dtype=np.float32
                )
            else:
                self.Maps[nm] = np.zeros([BatchSize, Hb, Wb], dtype=np.float32)
        # Start reading data multithreaded
        self.thread_list = []
        for pos in range(BatchSize):
            th = threading.Thread(
                target=self.LoadNext, name="threadReader" + str(pos), args=(pos, Hb, Wb)
            )
            self.thread_list.append(th)
            th.start()

    # Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
        '''
        This function waits until the data batch loading started at StartLoadBatch is finished

        '''
        for th in self.thread_list:
            th.join()

    def LoadBatch(self):
        '''
        This function loads the next batch of images

        Output:
            --Maps: dictionary of maps

        '''
        self.WaitLoadBatch()
        Maps = self.Maps

        self.StartLoadBatch()
        return Maps

    # Read single image annotation and data with no augmentation for testing
    def LoadSingle(self, MaxSize=1000):

        '''
        This function reads a single image annotation and data with no augmentation for testing

        Input:
            --MaxSize: maximum size of image

        Output:
            --Maps: dictionary of maps

        '''
        if self.itr >= len(self.AnnList):
            self.itr = 0
            self.epoch += 1

        Ann = self.AnnList[self.itr]
        self.itr += 1
        print(Ann["VesselWithContentRGB"])
        Maps = {}  # Annotaion maps

        with open(Ann["CameraParameters"]) as f:
            Maps["CameraParameters"] = json.load(f)

        for nm in MapsAndDepths:  # Read depth maps and segmentation mask
            if not nm in Ann:
                continue
            Path = Ann[nm]
            Depth = MapsAndDepths[nm]
            if ".exr" in Path:  #  read Depth and normals
                I = cv2.imread(Path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if np.ndim(I) >= 3 and Depth == 1:
                    I = I[:, :, 0]
            else:  # Read  masks and image
                if Depth == 1:
                    I = cv2.imread(Path, 0)
                else:
                    I = cv2.imread(Path)

            Maps[nm] = I.astype(np.float32)

        # Proccess  segmentation masks
        Maps["ContentRGB"] = cv2.imread(Ann["ContentRGB"])
        Maps["VesselMask"][Maps["VesselMask"] > 0] = 1
        Maps["VesselOpeningMask"] = (Maps["VesselOpening_Depth"] < 5000).astype(
            np.float32
        )
        Maps["ContentMaskClean"] = (Maps["ContentMask"].sum(2) > 0).astype(np.float32)
        Maps["ROI"] = np.ones(Maps["VesselMask"].shape, dtype=np.float32)

        IgnoreMask = Maps["ContentMask"][
            :, :, 2
        ]  # Undistorted content not viewed trough the vessel walls is ignored (leaking)
        IgnoreMask[
            Maps["ContentMask"][:, :, 1] > 0
        ] = 0  # Contet viewed trough vessel opening is not ignored
        IgnoreMask[
            (Maps["ContentMask"][:, :, 1] * Maps["ContentMask"][:, :, 0]) > 0
        ] = 1  # areas where the content is viewd trough the vessel floor are ignored
        Maps["ROI"][
            IgnoreMask > 0
        ] = 0  # Region of interest where the annotation is well defined

        # Generate depth maps
        Maps["EmptyVessel_Depth"][
            Maps["EmptyVessel_Depth"] > 5000
        ] = 0  # Remove far away background points
        Maps["VesselOpening_Depth"][
            Maps["VesselOpening_Depth"] > 5000
        ] = 0  # Remove far away background points
        Maps["ContentDepth"][
            Maps["ContentDepth"] > 5000
        ] = 0  # Remove far away background points

        # More changes to match specific key format expected by evaluating function
        del Maps["CameraParameters"]

        Maps["ContentMask"] = Maps["ContentMaskClean"]
        Maps["VesselXYZMask"] = Maps["VesselMask"].copy()
        Maps["ContentXYZMask"] = Maps["ContentMask"].copy()
        # Resize if too big
        h, w = Maps["VesselMask"].shape
        r = np.min([MaxSize / h, MaxSize / w])
        if r < 1:
            for nm in Maps:
                Maps[nm] = cv2.resize(
                    Maps[nm],
                    dsize=(int(r * w), (r * w)),
                    interpolation=cv2.INTER_NEAREST,
                )
        # Expand dimension to create batch like array expected by the net
        for nm in Maps:
            Maps[nm] = np.expand_dims(Maps[nm], axis=0)
        # Return
        return Maps



MapsAndDepths_LabPics = {
    "VesselMask": 1,  # Depth/Layers
    "VesselWithContentRGB": 3,
    "ContentMaskClean": 1,
    "ROI": 1,
}


class LabPics_Reader:
    """
    This class is used to read the data from the LabPics dataset

    MainDir: The main directory of the dataset
    MaxBatchSize: The maximum number of images in the batch
    MinSize: The minimum size of the image in pixels
    MaxSize: The maximum size of the image in pixels
    MaxPixels: The maximum number of pixels in the batch
    """

    # Initiate reader and define the main parameters for the data reader
    def __init__(
        self,
        MainDir=r"",
        MaxBatchSize=100,
        MinSize=250,
        MaxSize=1000,
        MaxPixels=800 * 800 * 5,
    ):
        self.MaxBatchSize = MaxBatchSize  # Max number of image in batch
        self.MinSize = MinSize  # Min image width and height in pixels
        self.MaxSize = MaxSize  # Max image width and height in pixels
        self.MaxPixels = MaxPixels  # Max number of pixel in all the batch (reduce to solve  out of memory issues)
        self.epoch = 0  # Training Epoch
        self.itr = 0  # Training iteratation
        self.AnnList = []  # Image/annotation list
        self.AnnByCat = {}  # Image/annotation list by class

        print("Creating annotation list for reader. This might take a while.")
        for AnnDir in os.listdir(MainDir):  # List of all example
            self.AnnList.append(MainDir + "/" + AnnDir)


        print(
            "Done making file list.\nTotal number of samples = "
            + str(len(self.AnnList))
        )

        self.StartLoadBatch()  # Start loading semantic maps batch (multi threaded)
        self.AnnData = False

    def GetNumSamples(self):
        '''
        This function returns the number of samples in the dataset
        '''

        return len(self.AnnList)

    def CropResize(self, Maps, Hb, Wb):
        '''
        This function crops and resizes the image and maps and ROI to fit the batch size

        Input:
            --Maps: dictionary of maps
            --Hb: height of batch
            --Wb: width of batch

        Output:
            --Maps: dictionary of maps

        '''
        h, w = Maps["ROI"].shape
        Bs = np.min((h / Hb, w / Wb))
        if (
            Bs < 1 or Bs > 3 or np.random.rand() < 0.2
        ):  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(h / Bs) + 1
            w = int(w / Bs) + 1
            for nm in Maps:
                if hasattr(Maps[nm], "shape"):  # check if array
                    if "RGB" in nm:
                        Maps[nm] = cv2.resize(
                            Maps[nm], dsize=(w, h), interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        Maps[nm] = cv2.resize(
                            Maps[nm], dsize=(w, h), interpolation=cv2.INTER_NEAREST
                        )

        if w > Wb:
            X0 = np.random.randint(w - Wb) 
        else:
            X0 = 0
        if h > Hb:
            Y0 = np.random.randint(h - Hb)  
        else:
            Y0 = 0

        for nm in Maps:
            if hasattr(Maps[nm], "shape"):  # check if array
                Maps[nm] = Maps[nm][Y0 : Y0 + Hb, X0 : X0 + Wb]

        for nm in Maps:
            if hasattr(Maps[nm], "shape"):  # check if array
                if not (Maps[nm].shape[0] == Hb and Maps[nm].shape[1] == Wb):
                    Maps[nm] = cv2.resize(
                        Maps[nm], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST
                    )

        return Maps

    def Augment(self, Maps):
        '''
        This function augments the image and mask

        It applies the following augmentations for 50% of the images:
            --Flip left right
            --Gaussian blur
            --Dark light
            --GreyScale

        Input:
            --Maps: dictionary of maps

        Output:
            --Maps: dictionary of augmented maps

        '''

        if np.random.rand() < 0.5:  # flip left right
            for nm in Maps:
                if hasattr(Maps[nm], "shape"):
                    Maps[nm] = np.fliplr(Maps[nm])
        for nm in Maps:
            if "RGB" in nm:
                if np.random.rand() < 0.1:  # Gaussian blur
                    Maps[nm] = cv2.GaussianBlur(Maps[nm], (5, 5), 0)

                if np.random.rand() < 0.1:  # Dark light
                    Maps[nm] = Maps[nm] * (0.5 + np.random.rand() * 0.65)
                    Maps[nm][Maps[nm] > 255] = 255

                if np.random.rand() < 0.1:  # GreyScale
                    Gr = Maps[nm].mean(axis=2)
                    r = np.random.rand()

                    Maps[nm][:, :, 0] = Maps[nm][:, :, 0] * r + Gr * (1 - r)
                    Maps[nm][:, :, 1] = Maps[nm][:, :, 1] * r + Gr * (1 - r)
                    Maps[nm][:, :, 2] = Maps[nm][:, :, 2] * r + Gr * (1 - r)

        return Maps

    def LoadNext(self, pos, Hb, Wb):
        '''
        This function reads the next image annotation and data

        Input:
            --pos: position in batch
            --Hb: height of batch
            --Wb: width of batch

        Output:
            --Maps: dictionary of maps

        '''
        AnnInd = np.random.randint(len(self.AnnList))

        InPath = self.AnnList[AnnInd]

        Img = cv2.imread(InPath + "/Image.jpg")  # Load Image
        if Img.ndim == 2:  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels in case there are more

        SemanticDir = InPath + r"/SemanticMaps/FullImage/"

        VesselMask = np.zeros(Img.shape)
        FilledMask = np.zeros(Img.shape)
        PartsMask = np.zeros(Img.shape)
        Ignore = np.zeros(Img.shape[0:2])
        MaterialScattered = np.zeros(Img.shape)

        if os.path.exists(SemanticDir + "//Transparent.png"):
            VesselMask = cv2.imread(SemanticDir + "//Transparent.png")
        if os.path.exists(SemanticDir + "//Filled.png"):
            FilledMask = cv2.imread(SemanticDir + "//Filled.png")
        if os.path.exists(SemanticDir + "//PartInsideVessel.png"):
            PartsMask = cv2.imread(SemanticDir + "//PartInsideVessel.png")
        if os.path.exists(SemanticDir + "//MaterialScattered.png"):
            MaterialScattered = cv2.imread(SemanticDir + "//MaterialScattered.png")
        if os.path.exists(InPath + "//Ignore.png"):
            Ignore = cv2.imread(InPath + "//Ignore.png", 0)

        Msk = {}

        Msk["VesselWithContentRGB"] = Img
        Msk["VesselMask"] = (VesselMask[:, :, 0] > 0).astype(np.float32)
        Msk["VesselMask"][PartsMask[:, :, 0] > 0] = 1
        Msk["ROI"] = (1 - Ignore).astype(np.float32)
        Msk["ROI"][FilledMask[:, :, 2] > 15] = 0
        Msk["ROI"][MaterialScattered[:, :, 2] > 0] = 0
        Msk["ContentMaskClean"] = (FilledMask[:, :, 0] > 0).astype(np.float32) * Msk[
            "VesselMask"
        ]

        Maps = self.Augment(Msk)
        if Hb != -1:
            Maps = self.CropResize(Maps, Hb, Wb)

        for nm in Maps:
            if nm in self.Maps:
                self.Maps[nm][pos] = Maps[nm]

    def StartLoadBatch(self):
        '''
        This function starts loading the next batch of images
            
        '''
        
        while True:
            Hb = np.random.randint(
                low=self.MinSize, high=self.MaxSize
            )  # Batch hight #900
            Wb = np.random.randint(
                low=self.MinSize, high=self.MaxSize
            )  # batch  width #900
            if Hb * Wb < self.MaxPixels:
                break
        BatchSize = np.int(
            np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize))
        )
        self.Maps = {}
        for nm in MapsAndDepths_LabPics:  # Create enoty
            if MapsAndDepths_LabPics[nm] > 1:
                self.Maps[nm] = np.zeros(
                    [BatchSize, Hb, Wb, MapsAndDepths_LabPics[nm]], dtype=np.float32
                )
            else:
                self.Maps[nm] = np.zeros([BatchSize, Hb, Wb], dtype=np.float32)
        self.thread_list = []
        for pos in range(BatchSize):
            th = threading.Thread(
                target=self.LoadNext, name="threadReader" + str(pos), args=(pos, Hb, Wb)
            )
            self.thread_list.append(th)
            th.start()

    def WaitLoadBatch(self):
        '''
        This function waits until the data batch loading started at StartLoadBatch is finished
            
        '''
        for th in self.thread_list:
            th.join()

    def LoadBatch(self):
        '''
        This function loads the next batch of images

        Output:
            --Maps: dictionary of maps

        '''
        self.WaitLoadBatch()
        Maps = self.Maps
        self.StartLoadBatch()
        return Maps
