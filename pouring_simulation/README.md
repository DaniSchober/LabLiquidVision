
# Flexible Pouring Simulation for Chemical Containers with Different Starting Volumes
 
[![NVIDIA Flex](https://img.shields.io/badge/NVIDIA-Flex-green)](https://developer.nvidia.com/flex)
![C++](https://img.shields.io/badge/C++-11-orange)

The aim of this part of the project is to be able to use the vision-based liquid volume prediction for the pouring container to subsequently take the results from the simulation to get the required robot movement to pour a specific amount of the predicted liquid volume into the target vessel. The idea is visualized in the following picture.

![Idea_Pouring](https://github.com/DaniSchober/LabLiquidVision/assets/75242605/22f34bab-c1ff-4e21-8b93-68bc24e62e57)

## Table of Contents

- [Basic Explanation](#explanation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Results & Videos](#results)
- [Project Organization](#orga)


## <a id="explanation"></a> Basic Explanation

The simulation is done in NVIDIA Flex. NVIDIA Flex is a GPU-based particle simulation library that allows realistic and interactive simulations of flexible and deformable materials. It allows to create of visually compelling effects such as fluid simulations, cloth simulations, and destruction effects in real-time. By harnessing the power of the GPU, NVIDIA Flex delivers high-performance and highly scalable simulations.

The provided code uses NVIDIA Flex to create a simulation of the pouring of a liquid from one container to another. Two containers used in cell culture processes were simulated. 
For each container, there is a separate scene header file, which defines the classes named 'Pouring_Flask' and 'Pouring_Bottle', which inherit from a base class called Scene.
These classes contain member variables and functions for simulating the pouring process. The 'Initialize' function initializes the simulation by creating the pouring and receiving containers, setting fluid parameters, and creating an emitter for the fluid particles. The 'Update' function is called every frame to update the simulation. The 'InPouringContainer' and 'InReceivingFlask' functions check if a particle is inside the pouring or receiving container, respectively. The results of each scene (TCP positions, parameters, and volume vs. theta) are saved in separate text files.

In the main file, the general simulation specifications are defined. Additionally, the code generates multiple pouring scenes with different parameters for flasks and bottles, as well as a calibration scene, and stores them in the g_scenes vector. The counts of flask and bottle scenes are printed at the end. The varied parameters are the starting volume in the pouring container, the maximum angle of the pouring, and the duration of the waiting time at the maximum angle. The rotation speed stays constant. The simulation uses the object files stored in the 'data' folder (receiving and pouring containers) and stores the results of each scene in the 'output' folder. The starting positions of the containers, as well as the size of the liquid emitter, can be modified in the configuration files of the 'data' folder.

![Pouring_Bottle](https://github.com/DaniSchober/LabLiquidVision/assets/75242605/98932a03-626c-491b-a0c8-84631a8608ad)

## Requirements

Supported Platforms:

* Windows 32/64 bit (CUDA, DX11, DX12)
* Linux 64 bit (CUDA, tested with Ubuntu 16.04 LTS and Mint 17.2 Rafaela)

A D3D11 capable graphics card with the following driver versions:

* NVIDIA GeForce Game Ready Driver 396.45 or above
* AMD Radeon Software Version 16.9.1 or above
* Intel® Graphics Version 15.33.43.4425 or above

To build the demo, at least one of the following is required:

* Microsoft Visual Studio 2013
* Microsoft Visual Studio 2015
* g++ 4.6.3 or higher

And either: 

* CUDA 9.2.148 toolkit
* DirectX 11/12 SDK

For further information about the installation, please refer to the [NVIDIA Flex repository](https://github.com/NVIDIAGameWorks/FleX).

## Usage
Make sure that the requirements are met. To start the implemented pouring simulation, just run one of the following from the command line:

```
run_cuda.bat
```
```
run_dx.bat
```

To change the simulation, mainly the `data` folder, the `demo/main.cpp` file, and the `demo/scenes/` files are important.

To change the pouring and/or receiving container, add the `.obj` file of the new container and a configuration file with the same name and the `.cfg` ending in the `data` folder. The config files contain the following information: 

1. TCP_x (inches)
2. TCP_y (inches)
3. Distance from TCP to CoR (inches)
4. Emitter diameter (inches)

The defined TCP is always relative to the origin defined in the `.obj` file.

To change the parameter space or the pouring object used in a specific scene, the scene creation in the `main.cpp` file needs to be changed.

The rest of the settings can be adapted directly in the specific scene files. The calibration scene (a 500 mL container for the particle/volume calculation) is defined in the `calibration.h` file. The pouring scene with the 400 mL cell culture flask is defined in `pouring_flask.h`, and the pouring with the Gibco 500 mL bottle in the `pouring_bottle.h` file. In these files, the liquid parameters, the trajectory of the pouring movement, and the particle counting can be changed.

To create a new .exe file after the changes from the Visual Studio project, you need to build the project in Release mode. Here's how you can create a .exe file:

1. Open the project in Visual Studio.
2. Ensure that the project configuration is set to "Release" and the platform is set to the appropriate target platform (x64).
3. Build the project by selecting "Build" > "Build Solution" from the menu or pressing Ctrl+Shift+B.
4. Once the build process completes successfully, the `.exe` file is saved in the `bin` folder and can be started using `run_cuda.bat` or `run_dx.bat`.

## <a id="results"></a> Results & Videos

### Demonstration of the simulation-to-reality transfer for the media/washing solution bottle (start volume: 130 mL, stop angle: 70°, stop time: 1.8s)

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/fdb614cc-5dbc-4781-afe1-ce0905711a20

### Demonstration of the simulation-to-reality transfer for the cell culture flask 400 mL (start volume: 30 mL, stop angle: 18°, stop time: 0.6s)

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/4e8628c3-f662-4428-8c60-f079eb56bbaf

### Demonstration of the simulation-to-reality experiment procedure of pouring from the media/washing solution bottle using a UR5e

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/6e451f05-4713-47eb-9a30-be0326475e82

### Simulations of the pouring with the cell culture flask from different camera angles

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/a2dd9ad1-54f2-40df-bba2-8e331427c311

### Simulations of the pouring with the media/washing solution bottle from different camera angles

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/81c2f8df-38dc-482b-856b-fb01ad717b06

## <a id="orga"></a> Project Organization

------------

    ├── LICENSE
    ├── README.md                  <- The top-level README for developers using this project.
    │
    ├── data                       <- Object files and configuration files for object import.
    │
    ├── demo                       <- Main code for the simulation.
    │   ├── main.cpp               <- Definition of the parameter space and creation of the required scenes.
    │   │
    │   ├── scenes                 <- Header files of the scenes.
    │   │   ├── calibration.h      <- Calibration scene using a 500 mL bucket.
    │   │   ├── pouring_bottle.h   <- Scene for pouring from a Gibco 500 mL bottle.
    │   │   └── pouring_flaks.h    <- Scene for pouring from a 400 mL cell culture flask.
    │   │   
    │   └── scenes.h               <- Includes the selected scenes.
    │
    ├── output                     <- Results of the simulation scenes.
    │   ├── CellFlask              <- Detailed results of the cell culture flask pouring scenes.
    │   │
    │   ├── MediumBottle           <- Detailed results of the media bottle pouring scenes.
    │   │
    │   └── Plots                  <- Plots of the analysis of the parameter space.
    │      
    ├── simulation_to_reality      <- Testing scripts and results of the simulation-to-reality transfer to the UR5e.
    │    
    ├── run_cuda.bat               <- Running the simulation using cuda.  
    │
    └── run_dx.bat                 <- Running the simulation using DX.  

--------









