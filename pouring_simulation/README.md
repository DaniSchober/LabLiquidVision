
# Flexible Pouring Simulation for Chemical Containers with Different Starting Volumes
 
[![NVIDIA Flex](https://img.shields.io/badge/NVIDIA-Flex-green)](https://developer.nvidia.com/flex)
![C++](https://img.shields.io/badge/C++-11-orange)

## Description

![Screenshot 2023-05-26 143110](https://github.com/DaniSchober/thesis/assets/75242605/45cbc248-175e-48ff-95da-4e0a2bf3932f)
Please write description here.

## Table of Contents

- [Supported Platforms](#platforms)
- [Requirements](#requirements)
- [Further Information](#information)
- [Usage](#usage)
- [Basic Explanation](#explanation)

## <a id="platforms"></a> Supported Platforms

* Windows 32/64 bit (CUDA, DX11, DX12)
* Linux 64 bit (CUDA, tested with Ubuntu 16.04 LTS and Mint 17.2 Rafaela)

## Requirements

A D3D11 capable graphics card with the following driver versions:

* NVIDIA GeForce Game Ready Driver 396.45 or above
* AMD Radeon Software Version 16.9.1 or above
* Intel® Graphics Version 15.33.43.4425 or above

To build the demo at least one of the following is required:

* Microsoft Visual Studio 2013
* Microsoft Visual Studio 2015
* g++ 4.6.3 or higher

And either: 

* CUDA 9.2.148 toolkit
* DirectX 11/12 SDK

## <a id="information"></a> Further Information

For further information about the installation please refer to the [NVIDIA Flex repository](https://github.com/NVIDIAGameWorks/FleX).

## Usage


## <a id="explanation"></a> Basic Explanation

The provided code is a C++ program that simulates the pouring of a liquid from one container to another. Two containers used in cell culture processes were used. 
For each container, there is a seperate scene header file, which defines the classes named 'Pouring_Flask' and 'Pouring_Bottle', which inherit from a base class called Scene.
These classes contain member variables and functions for simulating the pouring process. The 'Initialize' function initializes the simulation by creating the pouring and receiving containers, setting fluid parameters, and creating an emitter for the fluid particles. The 'Update' function is called every frame to update the simulation. The 'InPouringContainer' and 'InReceivingFlask' functions check if a particle is inside the pouring or receiving container, respectively. The results of each scene (TCP positions, parameters, and volume vs. theta) are saved in seperate text files.

In the main file, the general simulation specifications are defined. Additionally, the code generates multiple pouring scenes with different parameters for flasks and bottles, as well as a calibration scene, and stores them in the g_scenes vector. The counts of flask and bottle scenes are printed at the end. The varied parameters are the starting volume in the pouring container, the maximum angle of the pouring, and the duration of the waiting time at the maximum angle. The rotation speed stays constant.





