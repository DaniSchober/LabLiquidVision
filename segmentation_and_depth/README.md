Segmentation and Depth Estimation of Liquids and Transparent Vessels
==============================
The aim is to create **segmentation and depth maps of both transparent vessels and liquids present in research laboratories based on a single input image**. This can be used for process monitoring during experiments carried out in the laboratory and for automating flexible robotic tasks (e.g., grasping containers in an unknown environment).

The approach is based on the idea and data provided by Eppel et al. in [2020](https://github.com/sagieppel/LabPics-medical-Computer-vision-for-liquid-samples-in-hospitals-and-medical-lab-) and [2022](https://github.com/sagieppel/Predicting-3D-shape-of-liquid-and-objects-inside-transparent-vessels-as-XYZ-map-from-a-single-image). However, the model trained [here](https://github.com/sagieppel/Predicting-3D-shape-of-liquid-and-objects-inside-transparent-vessels-as-XYZ-map-from-a-single-image) tries to predict the XYZ map directly from an RGB input image. As suggested by the authors, an approach to predicting a depth map and then transforming it into XYZ coordinates using the known camera parameters is preferable over predicting the 3D model as an XYZ map. In autonomous systems in research laboratories, the camera does not change, and the camera parameters can be assumed to be known. Hence, the model from [Eppel et al.](https://github.com/sagieppel/Predicting-3D-shape-of-liquid-and-objects-inside-transparent-vessels-as-XYZ-map-from-a-single-image) is modified to predict the depth maps instead of the XYZ maps. Also, the provided model in this work is only trained specifically on liquid content rather than on object and liquid content. 

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
