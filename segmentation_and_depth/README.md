Segmentation and Depth Estimation of Liquids and Transparent Vessels
==============================
The aim is to create **segmentation and depth maps of both transparent vessels and liquids present in research laboratories based on a single input image**. This can be used for process monitoring during experiments carried out in the laboratory and for automating flexible robotic tasks (e.g., grasping containers in an unknown environment).

The approach is based on the idea and data provided by Eppel et al. in [2020](https://github.com/sagieppel/LabPics-medical-Computer-vision-for-liquid-samples-in-hospitals-and-medical-lab-) and [2022](https://github.com/sagieppel/Predicting-3D-shape-of-liquid-and-objects-inside-transparent-vessels-as-XYZ-map-from-a-single-image). However, the model trained [here](https://github.com/sagieppel/Predicting-3D-shape-of-liquid-and-objects-inside-transparent-vessels-as-XYZ-map-from-a-single-image) tries to predict the XYZ map directly from an RGB input image. As suggested by the authors, an approach to predicting a depth map and then transforming it into XYZ coordinates using the known camera parameters is preferable over predicting the 3D model as an XYZ map. In autonomous systems in research laboratories, the camera does not change, and the camera parameters can be assumed to be known. Hence, the model from [Eppel et al.](https://github.com/sagieppel/Predicting-3D-shape-of-liquid-and-objects-inside-transparent-vessels-as-XYZ-map-from-a-single-image) is modified to predict the depth maps instead of the XYZ maps. Also, the provided model in this work is only trained specifically on liquid content rather than on object and liquid content. 

## Installation

Please refer to the installation paragrah of the parent repository to create the required conda environment.

## Dataset 

The TransProteus dataset can be downloaded from [here](https://e.pcloud.link/publink/show?code=kZfx55Zx1GOrl4aUwXDrifAHUPSt7QUAIfV). To train the model, save the data in `data/interim`, or change the paths to the training folders in `src/data/make_dataset.py`. The LabPics data can be downloaded [here](https://zenodo.org/record/4736111). To get only the folders with liquid content and the required LabPics data, run `dvc pull` in this subfolder. 

## Usage

The main function takes the following arguments:

- `--mode`: Specify the mode (train, predict, or evaluate).
- `--cuda`: Set to `True` or `False` to enable or disable CUDA.
- `--model_path`: Path to the trained model.
- `--batch_size`: Batch size for training.
- `--epochs`: Number of epochs for training.
- `--load_model`: Set to `True` or `False` to load a pretrained model.
- `--use_labpics`: Set to `True` or `False` to use lab pictures for training.
- `--image_path`: Path to the image to predict.
- `--folder_path`: Path to the folder for evaluation.
 
 ### Training a model
 
 To train a model, get the data as explained above first. Change the paths to the training data folders in `src/data/make_dataset.py` for your folder structure.
 
 Example usage:
 ```
 python main.py --mode train --cuda True --batch_size 6 --epochs 75 --load_model False --use_labpics True
 ```
 
 The saved model gets saved in `models`, the parameters of the training in `logs`.
 
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
 - `Summary.png`: Summary picture containing the input image, and the predicted segmentation and depth maps.

![Summary](https://github.com/DaniSchober/thesis/assets/75242605/bba07710-c954-4ed0-9e41-69d154d6517a)

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── interim        <- 8 Transproteus folders and one LabPics folder.
    │
    ├── example            <- Location for example pictures used in prediction.
    │   └── results        <- Contains subfolders for the results of the prediction of images in example. 
    │
    ├── logs               <- Stored training parameters.
    │
    ├── models             <- Location where trained models are saved. Contains final model "segmentation_depth_model.torch"
    │
    ├── notebooks          <- Notebooks for testing of segmentation on a subset of self generated dataset, and plotting the loss.
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
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── main.py            <- Main file to train and evaluate models, and run inferences on images.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
