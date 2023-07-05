import src.data.load_data as DepthReader

"""
This file contains the function to create the readers for the training and testing data

The paths to the training and testing data and the max and min image size are defined here

"""

TransProteusFolder = {}
TransProteusFolder["Liquid1"] = r"data/interim/TranProteus1/Training/LiquidContent"
TransProteusFolder["Liquid2"] = r"data/interim/TranProteus2/Training/LiquidContent"
TransProteusFolder["Liquid3"] = r"data/interim/TranProteus3/Training/LiquidContent"
TransProteusFolder["Liquid4"] = r"data/interim/TranProteus4/Training/LiquidContent"
TransProteusFolder["Liquid5"] = r"data/interim/TranProteus5/Training/LiquidContent"
TransProteusFolder["Liquid6"] = r"data/interim/TranProteus6/Training/LiquidContent"
TransProteusFolder["Liquid7"] = r"data/interim/TranProteus7/Training/LiquidContent"
TransProteusFolder["Liquid8"] = r"data/interim/TranProteus8/Training/LiquidContent"

LabPicsFolder = {}
LabPicsFolder["LabPics"] = r"data/interim/LabPics Chemistry/Train"


MinSize = 270  # Min image dimension (height or width)
MaxSize = 1000  # Max image dimension (height or width)
MaxPixels = 800 * 800 * 2


def create_reader(MaxBatchSize):
    """
    This function creates the readers for the training data

    Input:
        --MaxBatchSize: maximum batch size

    Output:
        --Readers: dictionary of readers for each training folder

    """
    Readers = {}  # Transproteus readers
    for nm in TransProteusFolder:
        print("Folder used:", nm)
        Readers[nm] = DepthReader.Reader(
            TransProteusFolder[nm],
            MaxBatchSize,
            MinSize,
            MaxSize,
            MaxPixels,
            TrainingMode=True,
        )

    return Readers


def get_num_samples(Readers):
    """
    This function returns the number of samples in a dictionary of readers

    Input:
        --Readers: dictionary of readers for each training folder

    Output:
        --num_samples: number of samples in the training data

    """

    num_samples = 0
    for nm in Readers:
        num_samples += Readers[nm].GetNumSamples()
    return num_samples


def create_reader_LabPics(MaxBatchSize):
    """
    This function creates the readers for the LabPics training data

    Input:
        --MaxBatchSize: maximum batch size

    Output:
        --Readers: dictionary of readers for each training folder

    """

    Readers = {}
    for nm in LabPicsFolder:
        print("Folder used:", nm)
        Readers[nm] = DepthReader.LabPics_Reader(
            LabPicsFolder[nm],
            MaxBatchSize,
            MinSize,
            MaxSize,
            MaxPixels,
        )
    return Readers


def get_num_samples_LabPics(LabPics_Readers):
    """
    This function returns the number of samples in a dictionary of LabPics readers

    Input:
        --LabPics_Readers: dictionary of readers for each training folder

    Output:
        --num_samples: number of samples in the training data

    """
    num_samples = 0
    for nm in LabPics_Readers:
        num_samples += LabPics_Readers[nm].GetNumSamples()
    return num_samples


def create_reader_Test(MaxBatchSize, TestFolder):
    """
    This function creates the readers for the testing data

    Input:
        --MaxBatchSize: maximum batch size
        --TestFolder: path to testing data

    Output:
        --Readers: dictionary of readers for each testing folder

    """
    Readers = {}
    TestFolder_1 = {}
    TestFolder_1["Liquid1"] = TestFolder
    for nm in TestFolder_1:
        print("Folder used:", nm)
        Readers[nm] = DepthReader.Reader(
            TestFolder_1[nm],
            MaxBatchSize,
            MinSize,
            MaxSize,
            MaxPixels,
            TrainingMode=True,
        )

    return Readers
