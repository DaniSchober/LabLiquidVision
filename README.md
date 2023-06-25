# Computer vision-based detection and robotic handling of liquids for use in cell culture automation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NVIDIA Flex](https://img.shields.io/badge/NVIDIA-Flex-green)](https://developer.nvidia.com/flex)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![C++](https://img.shields.io/badge/C++-11-orange)

## Description

This work explores how computer vision-based liquid detection and handling can be integrated with robotic systems to develop a comprehensive and versatile automation system for cell culture applications for low throughput. The main idea of the project is visualized in [solution](#solution). It consists of four main subprojects that can be found in [subprojects](#subprojects).
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Subprojects](#subprojects)
- [Background](#background)
- [Overview of the Solution](#solution)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)
- [Contact](#contact)

## Installation

Please refer to the specific folders of the project for installation details for the specific use.

For a general setup using conda, do the following:

```python
git clone https://github.com/DaniSchober/LabLiquidVision
```
```python
conda env create -f environment.yml
```
```python
conda activate labliquidvision
```

## Usage

Here you can provide examples of how to use your project, including any command line arguments, sample inputs and outputs, etc.

## Subprojects

- [System Prototype for Cell Culture Automation](https://github.com/DaniSchober/thesis/tree/main/cell_culture_automation)
- [Vessel and Liquid Segmentation and Depth Estimation](https://github.com/DaniSchober/thesis/tree/main/segmentation_and_depth)
- [Liquid Volume Estimation](https://github.com/DaniSchober/thesis/tree/main/volume_estimation)
- [Pouring Simulation](https://github.com/DaniSchober/thesis/tree/main/pouring_simulation)


## Background

Cell culture is one of the fundamental tools in life science research and biotechnology. Applications range from testing of drugs or toxins, development of gene and cell therapies, and investigation of the function of biomolecules, to the production of biologics or vaccine particles. The focus lies on the problem of maintaining adherent cell cultures in laboratories. This includes two main procedures: (1) Media change and (2) passaging (subculturing) of cells. For adherent cell cultures, it is necessary to remove the spent media and replace it with fresh media repeatedly. In order to achieve optimal cell proliferation, it is essential to provide cells with fresh media two to three times per week and to maintain suitable conditions such as appropriate temperature, humidity, light, and pH. When cells reach confluency, it is crucial to subculture or passage them. Otherwise, they will experience reduced mitotic index and eventually result in cell death. The resulting cell suspension is then divided or reseeded into fresh cultures. The interval between cell passaging varies based on the cell line and its growth rate. The exact execution of the steps can vary depending on the researcher. For further visualization of the manual process, a video of the passaging process can be found here:

https://user-images.githubusercontent.com/75242605/235789312-c42b6f9f-35db-4578-a438-4b9f3dc0c1c3.mp4

## <a id="solution"></a> Overview of the Solution

The proposed solution consists of three main contributions, each taking a significant step towards building a fully autonomous cell culture system for low throughput. 
- **System prototype for full robotic automation:** The proposed system includes a UR5e collaborative industrial robot with an Intel RealSense D415 camera, a standard incubator that is adapted for automated opening and closing, a microscope, a unit for heating and cooling of liquids, and a capping and decapping unit. All the additional parts are 3D-printed or custom-built. The entire system is made to fit on one table. The robot arm executes the tasks usually done by human labor. The laboratory technician is only responsible for providing empty input flasks and refilling the required liquids. 
- **Vision-based transparent object detection and liquid volume estimation:** A vision-based system is developed to detect transparent objects for process monitoring (e.g. how many flasks are present), and to estimate the volume of liquid in the different transparent containers. The first step to achieve this is a deep-learning approach based on the [TransProteus dataset](https://www.cs.toronto.edu/matterlab/TransProteus/). The result of this model is a segmentation and depth estimation of transparent vessels and the liquid inside of them. This model is used to generate a new dataset of images of laboratory containers with liquid content, including their segmentation and depth estimation, and the object and liquid volumes. A new model trained on this dataset estimates the volume of liquid given in a transparent container. This estimation builds the base for autonomous robotic pouring from the given containers. 
- **Adaptable robotic pouring using fluid simulation:** Instead of pipetting, the autonomous solution is based on robotic pouring. The robot arm movement to pour a desired amount of liquid from a container with varying starting volumes is based on the vision-based estimation of the liquid volume, and the results of a simulation of the pouring movements with the particle-based simulator NVIDIA Flex. This pouring simulation can be adapted to different scenarios and objects also outside of chemical environments.

<img src="https://user-images.githubusercontent.com/75242605/236624972-99bd9f4b-c346-44b6-8f89-11d8068a09f8.png" width="700">

## Project Organization
------------

    ├── README.md                 <- The top-level README for developers using this project.
    │
    ├── cell_culture_automation   <- Subproject containing the code for robot movements and other system operations
    │
    ├── pouring_simulation        <- Subproject containing the simulation of pouring liquids with a UR5e in NVIDIA Flex
    │  
    ├── report                    <- Contains figures and results presented in the report.
    │   
    ├── segmentation_and_depth    <- Subproject for training, evaluating and using a CNN for segmentation and depth estimation for liquids and transparent containers from an RGB image.
    │
    ├── volume_estimation         <- Subproject for training, evaluating and using a CNN for liquid volume estimation based on an RGB image.
    │
    └── environment.yml           <- File for creating a conda environment with all required dependencies.
--------

## License

This project is licensed under the terms of the MIT license. 

## Credits

This project was developed in collaboration with [Novo Nordisk](https://www.novonordisk.com/), a global healthcare company that is dedicated to improving the lives of people with diabetes and other chronic diseases. Novo Nordisk provided support and resources for this project, including expert guidance, and access to equipment and facilities.

## Contact

For any inquiries or feedback, feel free to contact us:

- **Email:** danischober98@gmail.com
- **LinkedIn:** [Daniel Schober](https://www.linkedin.com/in/d-schober)
- **Website:** [Novo Nordisk](https://www.novonordisk.com/)
