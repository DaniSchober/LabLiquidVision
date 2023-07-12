Volume Estimation of Liquids in Transparent Vessels in Research Laboratory Environments
==============================
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

The aim is to **predict the volume of liquids inside transparent vessels** present in research laboratories based on a single RGB image. For the input of the volume estimation model, the described [segmentation and depth estimation network](https://github.com/DaniSchober/thesis/tree/main/segmentation_and_depth) is used first to create the segmented depth maps of vessels and liquids. A vision-based volume estimation can be used for various robotic tasks in- and outside of research laboratories. Exemplary use cases for research laboratories are the autonomous pouring of liquids based on a known starting volume or the control of liquid materials in a self-driving laboratory. Almost empty vessels could be detected and automatically replaced using a mobile robot. To the author's knowledge, no similar two-step approach for the vision-based estimation of liquid volume has been used.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Exemplary Output](#output)
- [Results](#results)
- [Project Organisation](#orga)

## Installation

Please refer to the installation paragraph of the parent repository to create the required conda environment.

## Dataset 

A new dataset called **LabLiquidVolume**, which contains 5451 images of liquids in laboratory containers, including their liquid volume, was created. The images were taken using an Intel RealSense D415 camera. The ground truth of the liquid volume was measured using a Mettler Toledo XSR2002S balance with an accuracy of ± 0.5 mL. Twelve of the most common research laboratory containers, including the consumables used in cell culture processes, were selected for the dataset. For the processed final dataset, the content of each sample is copied and extended by the output of the segmentation and depth estimation network using the RGB images as input. This includes the segmented liquid and vessel depth maps, the liquid and vessel segmentation masks (each as .png and .npy files), and the unsegmented depth maps of liquid and vessel. Additionally, a combined image is created for easier visualization of the samples, including the original RGB image, the normalized segmented liquid and vessel depth maps, and the liquid and vessel segmentation masks.

To get the initial and processed dataset and the trained models, run `dvc pull data` in this subfolder. 

An overview of the content of the **LabLiquidVolume** dataset can be seen here:

![Samples_LabLiquidVolume](https://github.com/DaniSchober/LabLiquidVision/assets/75242605/fd755746-a141-4903-8db5-0e50177b0714)

Further examples of the recorded RGB images that demonstrate the diversity of the dataset can be seen here:

![Overview_LabLiquidVolume](https://github.com/DaniSchober/LabLiquidVision/assets/75242605/0eb59423-b0f0-4e4f-a261-faf5465663ee)

## Usage

To get the two main models (`volume_model_no_vol.pth`, `volume_model_with_vol.pth`), run `dvc pull models` inside this subproject. The models get saved in `volume_estimation/models`.  

This repository contains code for training, predicting, and testing models for volume estimation of liquids and possibilities for new data generation and data conversion to include the segmentation and depth maps. The main file, `main.py`, handles these operations based on the provided arguments.

The `main.py` can be used with the following command:

```python
python main.py --mode <mode> [--image_path <image_path>] [--folder_path <folder_path>] [--num_epochs <num_epochs>] [--vessel_volume <vessel_volume>] [--use_vessel_volume] [--no_GPU] [--path_input <path_input>] [--path_output <path_output>] [--model_path <model_path>]
```

The available modes are:

- `train`: Trains a model. If `--use_vessel_volume` is used, a model that takes the volume of the vessel as an additional input is trained. Otherwise, only the segmented depth maps of liquid and vessel are used as input. 
- `predict`: Predicts the volume of liquid in an RGB image. The image can be changed by using `--image_path`. If `--use_vessel_volume` is used, the model, which includes the vessel volume input, is used. The vessel volume can be provided using `--vessel_volume`.  
- `test`: Tests the model. If `--use_vessel_volume` is used, the model that takes the volume of the vessel as an additional input is tested. Otherwise, only the model trained with the segmented depth maps of liquid and vessel is tested. The folder for testing can be changed using `--folder_path`.
- `record`: Records new samples for the dataset. A user interface is opened to simplify the data generation. The vessel name, liquid volume, and liquid color need to be provided by the user. Every sample gets saved in a new subfolder in `data/interim`.
- `convert`: Converts the generated data to the processed dataset. The path for the input data can be defined using `--path_input`, the path for the processed data using `--path_output`. The segmentation and depth maps are generated using the model defined using `--model_path`
- `predict_on_depth_maps`: Predicts the volume of liquid based on already generated segmented depth maps of liquid and vessel. A random sample from `data/processed` will be selected for the prediction.

Optional arguments:
- `--no_GPU`: Disables GPU usage for prediction.
- `--image_path`: Path to the image for prediction (default: "example/image.png").
- `--folder_path`: Path to the folder containing the converted LabLiquidVolume dataset (default: "data/processed/").
- `--use_vessel_volume`: If that is selected, the training, testing, and predictions are made using the volume of the vessel (in mL) as an additional input.
- `--vessel_volume`: Volume of the vessel as input for the prediction when `--use_vessel_volume` is used (default: 0).
- `--num_epochs`: Number of epochs to train for (default: 200).
- `--model_path`: Path to the model for dataset conversion (default: "../segmentation_and_depth/models/segmentation_and_depth.torch").
- `--path_input`: Path to the input dataset for conversion (default: "data/interim").
- `--path_output`: Path to the output dataset for conversion (default: "data/processed").

Please note that some modes require additional arguments. Make sure to provide the necessary arguments based on the chosen mode.

### Exemplary usage

**Training a model**:

```python
python main.py --mode train --folder_path data/processed --num_epochs 200
```

**Testing a model**:
```python
python main.py --mode test --folder_path data/processed
```

**Predicting the volume of liquid inside a transparent container in an RGB image without providing the vessel volume (PNG or JPEG format)**:
```python
python main.py --mode predict --image_path example/image.png --no_GPU
```

**Predicting the volume of liquid inside a transparent container in an RGB image WITH providing the vessel volume (PNG or JPEG format)**:
```python
python main.py --mode predict --image_path example/image.png --vessel_volume 100 --no_GPU --use_vessel_volume
```

**Predicting the volume of liquid inside a transparent container based on segmented depth maps of liquid and vessel**:
```python
python main.py --mode predict_on_depth_maps --folder_path data/processed
```

**Recording new data for the dataset (requires a connected Intel RealSense D415)**
```python
python main.py --mode record
```

**Converting the dataset to include the segmented depth maps of liquids and vessels**
```python
python main.py --mode convert --path_input data/interim --path_output data/processed --model_path ../segmentation_and_depth/models/segmentation_and_depth.torch
```

## <a id="output"></a> Exemplary output

An exemplary output of the model for the prediction based on a single RGB image  is visualized here. 

![RGBImagevisualize](https://github.com/DaniSchober/LabLiquidVision/assets/75242605/365fa1a4-873e-46a9-8d4e-363c79715c71)

## Results
The training processes for the liquid volume estimation were performed on a Tesla V100 GPU, which is available in the DTU HPC clusters. The final model was trained for 200 epochs using 90% of the **LabLiquidVolume** 
dataset. 10% of the training data is used for validation.  All testing results are based on the remaining 10% of the data (545 samples). The input images undergo a downsampling process as part of the preprocessing stage. 
The final models were trained for 200 epochs with a learning rate of 0.001, a batch size of 8, and a dropout rate of 0.2.

The results of the testing with the model without vessel volume input can be seen here:

<img src="https://github.com/DaniSchober/LabLiquidVision/assets/75242605/5c832a16-74ae-4dac-989a-7b6a1539a754" width="700">

The results of the testing with the model WITH vessel volume input can be seen here:

<img src="https://github.com/DaniSchober/LabLiquidVision/assets/75242605/95e6af0c-cf07-4d19-aa67-ff88c975c366" width="700">


## <a id="orga"></a> Project Organization

------------

    ├── LICENSE
    ├── README.md                  <- The top-level README for developers using this project.
    ├── data
    │   ├── interim                <- Intermediate data recorded with the camera.
    │   └── processed              <- The final canonical data set for modeling. Includes the segmented depth maps.
    │
    ├── example                    <- Exemplary RGB images to run predictions on.
    │
    ├── main.py                    <- Main entry point to execute tasks.
    │    
    ├── models                     <- Location of the saved models. Contains two models: (1) volume_model_no_vol.pth (2) volume_model_with_vol.pth
    │
    ├── notebooks                  <- Jupyter notebooks for plotting and data analysis.
    │
    ├── output                     <- Visualizations and results of training and testing.
    │
    ├── requirements.txt           <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py                   <- makes project pip installable (pip install -e .) so src can be imported.
    ├── src                        <- Source code for use in this project.
    │   ├── __init__.py            <- Makes src a Python module
    │   │
    │   ├── data                   <- Scripts to generate, convert, and load data.
    │   │   ├── dataloader.py      <- Dataloader for training and testing.
    │   │   ├── make_dataset.py    <- Script for converting the data using the segmentation and depth estimation model.
    │   │   └── record_data.py     <- Script for recording new data, including a simple user interface.
    │   │
    │   ├── models_1_no_vol        <- Scripts to train models without vessel volume input and then use trained models to make predictions. 
    │   │   ├── model.py           <- Definition of the model architecture.
    │   │   ├── predict_full_pipeline.py <- Running the liquid volume prediction from an RGB image.
    │   │   ├── predict_vol.py     <- Running the liquid volume prediction from two segmented depth maps.
    │   │   ├── test_model.py      <- Test the model performance on part of the dataset.
    │   │   ├── train_model.py     <- Definition of the training loop.
    │   │   └── validate_model.py  <- Validate the model during training.
    │   │
    │   └── models_2_input_vol     <- Scripts to train models with vessel volume input and then use trained models to make predictions. 
    │       ├── model.py           <- Definition of the model architecture.
    │       ├── predict_full_pipeline.py <- Running the liquid volume prediction from an RGB image.
    │       ├── predict_vol.py     <- Running the liquid volume prediction from two segmented depth maps.
    │       ├── test_model.py      <- Test the model performance on part of the dataset.
    │       ├── train_model.py     <- Definition of the training loop.
    │       └── validate_model.py  <- Validate the model during training.
    │
    ├── tox.ini                    <- tox file with settings for running tox; see tox.readthedocs.io 
    │
    └── Vessel_Selection.csv       <- Specifications of the objects in the dataset.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Data version control

To upload new data:
* dvc add data
* git add data.dvc .gitignore
* git commit -m "Description"
* git tag -a "v1.0" -m "data v1.0"
* dvc push

To get the data:
* git pull
* dvc pull
