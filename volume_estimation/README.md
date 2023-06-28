Volume Estimation of Liquids in Transparent Vessels in Research Laboratory Environments
==============================
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

The aim is to **predict the volume of liquids inside transparent vessels** present in research laboratories based on a single RGB image. For the input of the volume estimation model, the described [segmentation and depth estimation network](https://github.com/DaniSchober/thesis/tree/main/segmentation_and_depth) is used first to create the segmented depth maps of vessels and liquids. A vision-based volume estimation can be used for various robotic tasks in- and outside of research laboratories. Exemplary use cases for research laboratories are the autonomous pouring of liquids based on a known starting volume or the control of liquid materials in a self-driving laboratory. Almost empty vessels could be detected and automatically replaced using a mobile robot. To the author's knowledge, no similar two-step approach for the vision-based estimation of liquid volume has been used.

## Installation

Please refer to the installation paragrah of the parent repository to create the required conda environment.

## Dataset 

A new dataset called **LabLiquidVolume**, which contains 5451 images of liquids in laboratory containers including their liquid volume, was created. The images were taken using an Intel RealSense D415 camera. The ground truth of the liquid volume was measured using a Mettler Toledo XSR2002S balance with an accuracy of ± 0.5 mL. Twelve of the most common research laboratory containers, including the consumables used in cell culture processes, were selected for the dataset. For the processed final dataset, the content of each sample is copied and extended by the output of the segmentation and depth estimation network using the RGB images as input. This includes the segmented liquid and vessel depth maps, the liquid and vessel segmentation masks (each as .png and .npy files), and the unsegmented depth maps of liquid and vessel. Additionally, a combined image is created for easier visualization of the samples, including the original RGB image, the normalized segmented liquid and vessel depth maps, and the liquid and vessel segmentation masks.

To get initial and processed dataset and the trained models, run `dvc pull` in this subfolder. 

## Usage

This repository contains code for training, predicting, testing of models for volume estimation of liquids, and possibilities for new data generation, and data conversion to include the segmentation and depth maps. The main file, `main.py`, handles these operations based on the provided arguments.

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

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
