Segmentaion and Depth Estimation of Liquids and Transparent Vessels
==============================

Liquid Depth Estimation for use in Laboratories

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
