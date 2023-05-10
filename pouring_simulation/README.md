
# Flexible Pouring Simulation for Chemical Containers with Different Starting Volumes
 
[![NVIDIA Flex](https://img.shields.io/badge/NVIDIA-Flex-green)](https://developer.nvidia.com/flex)
![C++](https://img.shields.io/badge/C++-11-orange)

## Description

Please write description here.

## Table of Contents

- [Supported Platforms](#platforms)
- [Requirements](#requirements)
- [Further Information](#information)
- [Usage](#usage)
- [License](#license)
- [Credits](#credits)
- [Contact](#contact)

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

For further information please refer to the [NVIDIA Flex repository](https://github.com/NVIDIAGameWorks/FleX).

## Data version control

To upload new data:
* dvc add data
* git add data.dvc .gitignore
* git commit -m "Description"
* git tag -a "v1.0" -m "data v1.0"

To get the data:
* git pull
* dvc pull