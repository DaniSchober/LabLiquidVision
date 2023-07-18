# Cell Culture Automation: System Prototype

The aim was to integrate existing laboratory hardware for a **proof-of-concept prototype of an autonomous cell culture system**.

The proposed system includes a UR5e collaborative industrial robot with an Intel RealSense D415 camera, a standard incubator that is adapted for automated opening and closing, a microscope, a unit for heating and cooling liquids, and a capping and decapping unit. All the additional parts are 3D-printed or custom-built. The entire system is made to fit on one table. The robot arm executes the tasks usually done by human labor. The laboratory technician is only responsible for providing empty input flasks and refilling the required liquids. 

## Table of Contents

- [Hardware Setup](#hardwaresetup)
- [Connection and Configuration](#connection)
- [Process and Autonomous Workflow Description](#process)
- [Vision-based Process Monitoring](#processmonitoring)
- [Usage](#usage)
- [Autonomous Workflows](#workflows)
- [Videos of the Modules](#videosmodules)
- [Project Organisation](#orga)

## <a id="hardwaresetup"></a> Hardware Setup

An overview of the hardware setup can be seen here:

<img src="https://github.com/DaniSchober/LabLiquidVision/assets/75242605/455447e8-535a-40fa-aa36-86aaea8002ff" width="900">

(1) Automated incubator. (2) Microscope. (3) Trypsin unit. (4) Capper/Decapper. (5) Lid holders. (6) Flask storage. (7) Flask holder for pouring. (8) UR5e with 3D-printed gripper fingers. (9) Heating and Cooling of Liquids units.

The following external equipment is needed. The rest of the elements can be 3D printed using the files in the `CAD` folder.

|  Quantity  | Equipment                                       |
|:---------:|-------------------------------------------------|
|     3     | Festo linear/swivel clamp (8072624)            |
|     1     | Universal Robots UR5e                         |
|     1     | VWR INCU-Line ILCO 180 Premium, CO2 Incubator  |
|     1     | Disp-X Bottle Dispenser 1-10mL (30373528)      |
|     1     | RGI 100 Rotary Electric Gripper Jaw           |
|     1     | CytoSMART Lux3 BR                             |
|     1     | Festo ISO cylinder DSNU-25-500-PPV-A           |
|     1     | Festo Compressed air filter regulator LFR (8003637) |
|     1     | Vention table (121 x 198 cm)                  |
|     1     | TCube Edge Thermoelectric Chiller (52-14072-1) |
|     2     | Festo Solenoid Valve (VUVS-LK20-B52-D-G18-1C1-S) |
|     1     | Festo Basic valve LR-D-MINI (546430)          |
|     1     | Robotiq Hand-E Adaptive Gripper               |

## <a id="connection"></a> Connection and Configuration

All the devices are either connected to the UR control box or directly to the PC (Dell Precision 3630 Tower). The connection between the UR5e and its control box and the PC is established by using the UR-RTDE library developed by SDU \cite{SDU2023UniversalInterface}. It provides a real-time interface to exchange I/O data and control the robot arm from an external application or program. The library works by establishing a communication link between the computer and the UR robot arm using an Ethernet connection. It utilizes the Real-Time Data Exchange (RTDE) protocol, which is a proprietary protocol developed by Universal Robots for real-time communication. The pneumatic clamps and the cylinder are controlled by using electrically actuated 5/2-way control valves. These are both connected to two digital output connectors of the UR control box. Each output can control the airflow in one output 6 mm tube of the valve, which is responsible for extending or compressing the clamps and the cylinder. The Robotiq Hand-E gripper is connected to the robot arm by a power and communication cable with a USB adapter. The microscope, camera, and rotary electric gripper are all connected by a USB cable (type 3.0 output) to the PC. While the camera and the gripper can be controlled by using specific Python libraries (pyrealsense2, minimalmodbus), the microscope needs the CytoSMART driver installed on the PC. The entire system is controlled by an application written in Python.

The functional interconnection of hard- and software is visualized here:

<img src="https://github.com/DaniSchober/LabLiquidVision/assets/75242605/258beef8-be43-4a33-b67d-bf682ef82842" width="700">

## <a id="process"></a>  Process and Autonomous Workflow Description

The autonomous process of the cell culture system is split into three high-level workflows:

- Workflow A: Analyzing cell growth.
- Workflow B: Changing media.
- Workflow C: Passaging.

The system is designed to execute these workflows independently, based on user input or a pre-scheduled plan. To enhance reusability and simplicity, the completion steps for each workflow are structured into modular components.

### Workflow Modules

The following provides a mid-level overview of the purpose of each process module, along with summarized descriptions of the substeps.

1. **Module 1: Get a cell culture flask from the incubator**
   - Open clamps and door.
   - Grip the flask.
   - Move the flask outside.
   - Close door and clamps.

2. **Module 2: Analyze cells**
   - Move the flask to the regripping station.
   - Regrip the flask.
   - Place the flask on the microscope.
   - Take images at different positions.
   - Move back to the start position.

3. **Module 3: Place a flask in a flask holder**
   - Move to the flask holder.
   - Place the flask inside.
   - Move back to the start position.

4. **Module 4: Decap flask(s)**
   - Move to the flask.
   - Grip the lid.
   - Take off the lid.
   - Place the lid on a lid holder.
   - Move back to the start position.

5. **Module 5: Remove liquid from a flask**
   - Take the open flask from a flask holder.
   - Move to the waste container.
   - Pour out all the liquid.
   - Place the flask back into the flask holder.

6. **Module 6: Add liquid to a flask**
   - Heat up media or washing solution.
   - Take the bottle.
   - Decap the bottle.
   - Regrip the bottle.
   - Pour a specific amount into the flask.
   - Regrip the bottle.
   - Cap the bottle.
   - Place it back into the bottle holder.

7. **Module 7: Add trypsin to a flask**
   - Move to the trypsin unit.
   - Move up the bottle dispenser.
   - Push it down.
   - Move back to the start position.

8. **Module 8: Get three empty flasks and place them into the flask holders**
   - Move to flask storage.
   - Grip an empty flask.
   - Move to the flask holder.
   - Place the flask inside.
   - Move back to the start position (3 times).

9. **Module 9: Split cells into empty flasks**
   - Take the full flask from the flask holder.
   - Move to the empty flasks.
   - Pour liquid three times.
   - Place the empty flask into the flask storage.

10. **Module 10: Cap flask(s)**
    - Get a lid from the lid holder.
    - Place the lid on a flask.
    - Cap the flask.
    - Move back to the start position.

11. **Module 11: Place a flask in the incubator**
    - Open clamps and door.
    - Move the flask inside.
    - Place the flask on the flask storage.
    - Close the door and clamps.

### Process Diagram

Throughout the implementation of the modules, careful consideration was given to the selection of start and end positions, ensuring the flexibility to arrange them in any order, thus enabling diverse workflows to be executed. The following shows the process diagram for the three primary workflows:

<img src="https://github.com/DaniSchober/LabLiquidVision/assets/75242605/4768f479-717a-4efb-a50c-4f3e59655297" width="800">

## <a id="processmonitoring"></a>  Deep learning-based process monitoring using the segmentation and depth estimation model

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/f8f00ed9-c70c-4642-8393-2c0ec90c2700

## Usage

To start the system, make sure all devices are physically connected and the UR5e is set to remote control, and then run the following command:

```python
python main.py
```

This command will launch the following graphical user interface (GUI). Before proceeding, make sure to connect the devices by selecting the "Connect Devices" button. Once connected, you can choose the desired cell culture process from the available options and select the flask number. The system will capture images from the camera and microscope, and store them for further analysis.

<img src="https://github.com/DaniSchober/LabLiquidVision/assets/75242605/a482ab19-30c4-443f-9a26-8a548b1e057e" width="800">

### Implementation Details

The `main.py` file serves as the central component of the Cell Culture Automation User Interface. It incorporates various modules and functionalities to facilitate the control and monitoring of the automated cell culture system. The following are the key implementation details of the `main.py` file:

1. **Importing Required Modules**: The necessary modules are imported at the beginning of the file to enable the functionality of the User Interface. These include `tkinter` for creating the graphical user interface, `PIL` for image processing, `glob` for file operations, `os` for operating system-related tasks, and other custom modules specific to the system's components.

2. **Device Connection**: The `connect_device()` function is responsible for establishing connections with the robotic arm, microscope, and decapper devices. It utilizes the respective modules for each device to establish the connections and displays a connection status message box to inform the user.

3. **Analyze Cell Growth**: The `analyze_cell_growth()` function prompts the user to input the flask number in the incubator to initiate the cell growth analysis process. It calls various functions to manipulate the robotic arm, microscope, and decapper devices accordingly. The captured images are resized and displayed in separate image windows.

4. **Change Media**: The `change_media()` function enables the user to change the media of a specific flask. Similar to the `analyze_cell_growth()` function, it prompts the user for the flask number, performs the required actions using the devices, captures and displays images, and executes the necessary steps for media change.

5. **Passage**: The `passage()` function allows the user to perform the cell passage process. After inputting the flask number, the function carries out the required operations using the devices, captures images, executes the steps for flask decapping, cell splitting, and flask recapping, and places the passaged flask back into the incubator.

6. **Graphical User Interface (GUI)**: The main window of the user interface is created using `tkinter`. The window includes buttons for connecting devices, analyzing cell growth, changing media, and performing passage. Additionally, separate frames are created to display captured images of cell growth analysis.


## <a id="workflows"></a>  Autonomous Workflows

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/94feda8f-667b-4c8b-9d64-9a755cea9c25

## <a id="videosmodules"></a> Videos of the modules

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/8ba1a7e2-e1f4-48b2-a36e-5c0b5226792f

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/da994fc4-18d3-4022-9db4-272dc1d39821

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/3c8aa1a7-f140-4d4f-b89e-7994704447a6

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/db8c39dd-f67e-4e6d-a76c-f8cda058b10c

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/bf03fc5d-229b-4272-8fcf-577293d30aec

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/6d8a44fc-060b-4510-bd73-011f697f86c2

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/def5a9d2-dd8b-42ec-a7f5-9e586546134d

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/62639b81-ccff-4570-8bc8-da9e96ec6865

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/dd90355e-74b1-4ca3-a029-94100849f5bc

https://github.com/DaniSchober/LabLiquidVision/assets/75242605/d8d92095-d140-4865-a37c-1243067234de

## <a id="orga"></a> Project Organization
------------

    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    │
    ├── CAD                    <- Location of the STL and IPT files and drawings of the 3D printed parts.
    │
    ├── captured_images        <- Stored images captured during the process.
    │
    ├── main                   <- Main folder containing the functions for all modules and workflows.
    │   ├── Cell_Flask_diff    <- Trajectories of the different simulated pouring movements with the cell culture flask.
    │   ├── Media_Bottle_diff  <- Trajectories of the different simulated pouring movements with the media bottle.
    │   ├── main.py            <- Main file for creating the GUI and starting the workflows.
    │   ├── rotation_matrix.py <- Functions for converting the pouring movement to the robot base coordinate system.
    │   └── utils.py           <- Functions to control robot and devices to execute the workflows.
    │
    └── notebooks              <- Notebooks for testing the autonomous pouring, the simulation-to-reality transfers and the process monitoring.

--------





