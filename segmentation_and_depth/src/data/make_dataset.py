import src.data.load_data as DepthReader

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
MaxPixels = (
    800 * 800 * 2
)  # Max pixels in a batch (not in image), reduce to solve out if memory problems
#MaxBatchSize = 6  # Max images in batch

def create_reader(MaxBatchSize):
    Readers = {}  # Transproteus readers
    for nm in TransProteusFolder:
        print("Folder used:", nm)
        #print(TransProteusFolder[nm])
        Readers[nm] = DepthReader.Reader(
            TransProteusFolder[nm],
            MaxBatchSize,
            MinSize,
            MaxSize,
            MaxPixels,
            TrainingMode=True,
        )

    #print("Readers:", Readers)
    return Readers

def get_num_samples(Readers):
    num_samples = 0
    for nm in Readers:
        num_samples += Readers[nm].GetNumSamples()
    return num_samples


def create_reader_LabPics(MaxBatchSize):
    Readers = {}  # Transproteus readers
    for nm in LabPicsFolder:
        print("Folder used:", nm)
        #print(TransProteusFolder[nm])
        Readers[nm] = DepthReader.LabPics_Reader(
            LabPicsFolder[nm],
            MaxBatchSize,
            MinSize,
            MaxSize,
            MaxPixels,
            #TrainingMode=True,
        )

    #print("Readers:", Readers)
    return Readers

def get_num_samples_LabPics(LabPics_Readers):
    num_samples = 0
    for nm in LabPics_Readers:
        num_samples += LabPics_Readers[nm].GetNumSamples()
    return num_samples


def create_reader_Test(MaxBatchSize, TestFolder):
    Readers = {}  # Transproteus readers
    TestFolder_1 = {}
    TestFolder_1["Liquid1"] = TestFolder
    for nm in TestFolder_1:
        print("Folder used:", nm)
        #print(TransProteusFolder[nm])
        Readers[nm] = DepthReader.Reader(
            TestFolder_1[nm],
            MaxBatchSize,
            MinSize,
            MaxSize,
            MaxPixels,
            TrainingMode=True,
        )

    #print("Readers:", Readers)
    return Readers