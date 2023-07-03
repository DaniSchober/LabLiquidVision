# Cell Culture Automation: System Prototype

The aim was to integrate existing laboratory hardware for a **proof-of-concept prototype of an autonomous cell culture system**.

The proposed system includes a UR5e collaborative industrial robot with an Intel RealSense D415 camera, a standard incubator that is adapted for automated opening and closing, a microscope, a unit for heating and cooling of liquids, and a capping and decapping unit. All the additional parts are 3D-printed or custom-built. The entire system is made to fit on one table. The robot arm executes the tasks usually done by human labor. The laboratory technician is only responsible for providing empty input flasks and refilling the required liquids. 

## Hardware Setup

An overview of the hardware setup can be seen here:

![System_Overview](https://github.com/DaniSchober/LabLiquidVision/assets/75242605/455447e8-535a-40fa-aa36-86aaea8002ff)

(1) Automated incubator. (2) Microscope. (3) Trypsin unit. (4) Capper/Decapper. (5) Lid holders. (6) Flask storage. (7) Flask holder for pouring. (8) UR5e with 3D-printed gripper fingers. (9) Heating and Cooling of Liquids Unit.

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

## Connection and Configuration

All the devices are either connected to the UR control box or directly to the PC (Dell Precision 3630 Tower). The connection between the UR5e and its control box and the PC is established by using the UR-RTDE library developed by SDU \cite{SDU2023UniversalInterface}. It provides a real-time interface to exchange I/O data and control the robot arm from an external application or program. The library works by establishing a communication link between the computer and the UR robot arm using an Ethernet connection. It utilizes the Real-Time Data Exchange (RTDE) protocol, which is a proprietary protocol developed by Universal Robots for real-time communication. The pneumatic clamps and the cylinder are controlled by using electrically actuated 5/2-way control valves. These are both connected to two digital output connectors of the UR control box. Each output can control the airflow in one output 6 mm tube of the valve, which is responsible for extending or compressing the clamps and the cylinder. The Robotiq Hand-E gripper is connected to the robot arm by a power and communication cable with a USB adapter. The microscope, camera, and rotary electric gripper are all connected by a USB cable (type 3.0 output) to the PC. While the camera and the gripper can be controlled by using specific Python libraries (pyrealsense2, minimalmodbus), the microscope needs the CytoSMART driver installed on the PC. The entire system is controlled by an application written in Python.

The functional interconnection of hard- and software is visualized in here:

![Connection_Setup](https://github.com/DaniSchober/LabLiquidVision/assets/75242605/258beef8-be43-4a33-b67d-bf682ef82842)

