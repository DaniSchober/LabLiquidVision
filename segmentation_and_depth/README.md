Segmentation and Depth Estimation of Liquids and Transparent Vessels
==============================
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

The aim is to create **segmentation and depth maps of both transparent vessels and liquids present in research laboratories based on a single input image**. This can be used for process monitoring during experiments carried out in the laboratory and for automating flexible robotic tasks (e.g., grasping containers in an unknown environment).

The approach is based on the idea and data provided by Eppel et al. in [2020](https://github.com/sagieppel/LabPics-medical-Computer-vision-for-liquid-samples-in-hospitals-and-medical-lab-) and [2022](https://github.com/sagieppel/Predicting-3D-shape-of-liquid-and-objects-inside-transparent-vessels-as-XYZ-map-from-a-single-image). However, the model trained [here](https://github.com/sagieppel/Predicting-3D-shape-of-liquid-and-objects-inside-transparent-vessels-as-XYZ-map-from-a-single-image) tries to predict the XYZ map directly from an RGB input image. As suggested by the authors, an approach to predicting a depth map and then transforming it into XYZ coordinates using the known camera parameters is preferable over predicting the 3D model as an XYZ map. In autonomous systems in research laboratories, the camera does not change, and the camera parameters can be assumed to be known. Hence, the model from [Eppel et al.](https://github.com/sagieppel/Predicting-3D-shape-of-liquid-and-objects-inside-transparent-vessels-as-XYZ-map-from-a-single-image) is modified to predict the depth maps instead of the XYZ maps. Also, the provided model in this work is only trained specifically on liquid content rather than on object and liquid content. 

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Point Cloud Visualization](#visual)
- [Project Organisation](#orga)

## Installation

Please refer to the installation paragraph of the parent repository to create the required conda environment.

## Dataset 

The TransProteus dataset can be downloaded from [here](https://icedrive.net/s/6cZbP5dkNG). To train the model, save the data in `data/interim`, or change the paths to the training folders in `src/data/make_dataset.py`. The LabPics data can be downloaded [here](https://zenodo.org/record/4736111). To get only the folders with liquid content and the required LabPics data, run `dvc pull` in this subfolder. 

## Usage

To get the trained model, run `dcv pull models` inside this folder. The model called `segmentation_and_depth_model.torch` will be saved in the `models` folder.

The main function takes the following arguments:

- `--mode`: Specify the mode (train, predict, or evaluate).
- `--cuda`: Set to `True` or `False` to enable or disable CUDA.
- `--model_path`: Path to the trained model.
- `--batch_size`: Batch size for training.
- `--epochs`: Number of epochs for training.
- `--load_model`: Set to `True` or `False` to load a pre-trained model.
- `--use_labpics`: Set to `True` or `False` to use lab pictures for training.
- `--image_path`: Path to the image to predict.
- `--folder_path`: Path to the folder for evaluation.
 
 ### Training a model
 
 To train a model, get the data as explained above first. Change the paths to the training data folders in `src/data/make_dataset.py` for your folder structure.
 
 Example usage:
 ```
 python main.py --mode train --cuda True --batch_size 6 --epochs 75 --load_model False --use_labpics True
 ```
 
 The saved model gets saved in `models`, and the parameters of the training in `logs`.
 
 ### Evaluating a model
 Example usage:
 ```
 python main.py --mode evaluate --cuda True --model_path models/segmentation_depth_model.torch --folder_path data/interim/TranProteus1/Testing/LiquidContent
 ```
 
 The results of the testing get saved in `output/results.txt`.
 
 ### Predict segmentation and depth maps on an image using a trained model
 Example usage:
 ```
 python main.py --mode predict --cuda True --model_path models/segmentation_depth_model.torch --image_path example/RGBImage.png
 ```
 
 The results of the prediction get saved in a subfolder in `example/results`. The output contains 8 images:
 - `Original Image.png`: Input image.
 - `ContentMaskClean.png`: Segmentation mask of the liquid.
 - `VesselMask.png`: Segmentation mask of the transparent vessel.
 - `VesselOpeningMask.png`: Segmentation mask of the opening of the vessel.
 - `ContentDepth.png`: Normalized depth map of the liquid content.
 - `EmptyVessel_Depth`: Normalized depth map of the transparent vessel.
 - `VesselOpening_Depth.png`: Normalized depth map of the opening of the vessel.
 - `Summary.png`: Summary picture containing the input image and the predicted segmentation and depth maps.

An exemplary output of the model can be seen here:

![Summary](https://github.com/DaniSchober/thesis/assets/75242605/bba07710-c954-4ed0-9e41-69d154d6517a)

## <a id="visual"></a> Point Cloud Visualization

The following presents a visualization of the point clouds created from the segmented depth maps from the model using the camera intrinsics from the Intel RealSense D415. The `pyrealsense2` library was used to get the camera intrinsics.

Input Image:

![Input_RGBImage](https://github.com/DaniSchober/thesis/assets/75242605/6be3f2ac-d6e1-4cfa-ad8b-b822920e8090)

Generated Point Clouds:

https://github.com/DaniSchober/thesis/assets/75242605/e4bd52c9-f5dd-4eaf-b8d0-144fe9be176d

https://github.com/DaniSchober/thesis/assets/75242605/e788d492-75cf-4c39-979b-9f43dbe01241


## <a id="orga"></a> Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── interim        <- 8 Transproteus folders and one LabPics folder.
    │
    ├── example            <- Location for pictures used in prediction.
    │   └── results        <- Contains subfolders for the results of the prediction of images in the example. 
    │
    ├── logs               <- Stored training parameters.
    │
    ├── models             <- Location where trained models are saved. Contains final model "segmentation_depth_model.torch"
    │
    ├── notebooks          <- Notebooks for testing of segmentation on a subset of the self-generated dataset, plotting the loss, and visualizing the point clouds.
    │    
    ├── output             <- Results of model evaluation.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate data and create the readers for the training and testing data
    │   │   ├── load_data.py    
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions as well as evaluating it
    │   │   ├── evaluate_model.py
    │   │   ├── loss_functions.py
    │   │   ├── model.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │    
    │   ├── utils          <- contains utils for model evaluation
    │   │
    │   └── visualization  <- Scripts to create exploratory and results-oriented visualizations
    │       └── visualize.py
    │
    ├── main.py            <- Main file to train and evaluate models and run inferences on images.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
