
# Flexible Pouring Simulation for Chemical Containers with Different Starting Volumes
 
[![NVIDIA Flex](https://img.shields.io/badge/NVIDIA-Flex-green)](https://developer.nvidia.com/flex)
![C++](https://img.shields.io/badge/C++-11-orange)

The aim of this part of the project is to be able to use the vision-based liquid volume prediction for the pouring container to subsequently take the results from the simulation to get the required robot movement to pour a specific amount of the predicted liquid volume into the target vessel. The idea is visualized in the following picture.

![Idea_Pouring](https://github.com/DaniSchober/LabLiquidVision/assets/75242605/22f34bab-c1ff-4e21-8b93-68bc24e62e57)


## Description

![Screenshot 2023-05-26 143110](https://github.com/DaniSchober/thesis/assets/75242605/45cbc248-175e-48ff-95da-4e0a2bf3932f)
Please write description here.

## Table of Contents

- [Basic Explanation](#explanation)
- [Supported Platforms](#platforms)
- [Requirements](#requirements)
- [Further Information](#information)
- [Usage](#usage)
- [Results](#results)


## <a id="explanation"></a> Basic Explanation

The simulation is done in NVIDIA Flex. NVIDIA Flex is a GPU-based particle simulation library that allows realistic and interactive simulations of flexible and deformable materials. It allows to create of visually compelling effects such as fluid simulations, cloth simulations, and destruction effects in real-time. By harnessing the power of the GPU, NVIDIA Flex delivers high-performance and highly scalable simulations.

The provided code uses NVIDIA Flex to create a simulation of the pouring of a liquid from one container to another. Two containers used in cell culture processes were simulated. 
For each container, there is a separate scene header file, which defines the classes named 'Pouring_Flask' and 'Pouring_Bottle', which inherit from a base class called Scene.
These classes contain member variables and functions for simulating the pouring process. The 'Initialize' function initializes the simulation by creating the pouring and receiving containers, setting fluid parameters, and creating an emitter for the fluid particles. The 'Update' function is called every frame to update the simulation. The 'InPouringContainer' and 'InReceivingFlask' functions check if a particle is inside the pouring or receiving container, respectively. The results of each scene (TCP positions, parameters, and volume vs. theta) are saved in separate text files.

In the main file, the general simulation specifications are defined. Additionally, the code generates multiple pouring scenes with different parameters for flasks and bottles, as well as a calibration scene, and stores them in the g_scenes vector. The counts of flask and bottle scenes are printed at the end. The varied parameters are the starting volume in the pouring container, the maximum angle of the pouring, and the duration of the waiting time at the maximum angle. The rotation speed stays constant. The simulation uses the object files stored in the 'data' folder (receiving and pouring containers) and stores the results of each scene in the 'output' folder. The starting positions of the containers, as well as the size of the liquid emitter, can be modified in the configuration files of the 'data' folder.

## Usage
Make sure that the requirements are met. To start the implemented pouring simulation, just run one of the following from the command line:

```
run_cuda.bat
```
```
run_dx.bat
```

To change the simulation, mainly the 

To create a .exe file from a Visual Studio project, you need to build the project in Release mode. By default, Visual Studio generates the executable file in the project's output directory, which is typically the "bin\Release" or "bin\x64\Release" folder. Here's how you can create a .exe file:

1. Open your project in Visual Studio.
2. Ensure that the project configuration is set to "Release" and the platform is set to the appropriate target platform (e.g., x86, x64).
3. Build the project by selecting "Build" > "Build Solution" from the menu or pressing Ctrl+Shift+B.
4. Once the build process completes successfully, navigate to the output directory of your project. You can find this directory in the Solution Explorer under the project node or by browsing to the project's folder on your file system.
5. Look for the generated .exe file within the output directory. The name of the .exe file will typically match the project name or the name specified in the project settings.
6. You can now distribute and run the generated .exe file on the target machine.

## <a id="platforms"></a> Supported Platforms

* Windows 32/64 bit (CUDA, DX11, DX12)
* Linux 64 bit (CUDA, tested with Ubuntu 16.04 LTS and Mint 17.2 Rafaela)

## Requirements

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

## <a id="information"></a> Further Information

For further information about the installation, please refer to the [NVIDIA Flex repository](https://github.com/NVIDIAGameWorks/FleX).

## Results

Demonstration of the simulation-to-reality experiment procedure of pouring from the media/washing solution bottle using a UR5e:

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/2282241b-7776-4f27-8715-334d02c7aebd









