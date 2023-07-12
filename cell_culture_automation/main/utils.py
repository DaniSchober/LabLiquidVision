# Import CytoSmart & useful modules
from typing import NamedTuple
from sila2.client import SilaClient as Client
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Import Robot module
import rtde_control
import rtde_receive
import rtde_io
from robotiq_gripper_control import RobotiqGripper
import time
import rotation_matrix as rm

# Import Decapper Module
import minimalmodbus
import threading

# Import Cooling System Module
from serial import Serial

# Import Robot positions
from positions import *

def connect_robot(ip = "192.168.2.1"):
    rtde_c = rtde_control.RTDEControlInterface(ip) #IP address found on robot
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    rtde_io_set = rtde_io.RTDEIOInterface(ip)
    print("Robot Connected")

    print("Activating Gripper")
    gripper = RobotiqGripper(rtde_c)
    gripper.activate()  # returns to previous position after activation
    gripper.set_force(20)  # from 0 to 100 %
    gripper.set_speed(20)  # from 0 to 100 %
    print("Gripper activated")
    return rtde_c, rtde_r, rtde_io_set, gripper

def connect_microscope():
    # Set Up CytoSmart Microscope
    settings = {
    "address": "localhost",
    "port": 50052,
    "insecure": True
    }

    client = Client(**settings)

    # List of all connected devices
    # All microcopes connected to this computer
    A = client.DeviceManagerService.Devices.get()
    serial = A[0].SerialNumber  # Get serial number of the microscope
    print(f"Serial: {serial}")

    # CytoSmart Take Picture
    class Input(NamedTuple):
        SerialNumber: str
        Group: str

    class Input2(NamedTuple):
        ImagePath: str
    return client, serial, Input, Input2

def create_directory(folder_name = "Cell_Confluency", path = "C:/Users/Operator/Desktop/Cell Images"):
    # Define the format of the directory name
    directory_name_format = "%Y-%m-%d_%H-%M-%S"
            
    # Get the current date and time in the desired format
    current_time = datetime.now().strftime(directory_name_format)
    folder_name = folder_name+'_'+current_time
            
    # Combine the path and directory name to create the full directory path
    full_path = os.path.join(path, folder_name)
            
    # Use the os.makedirs() function to create the directory
    os.makedirs(full_path)
            
    print(f"Directory '{folder_name}' created in '{path}'.")
            
    return full_path

def take_picture(path, client, serial, Input, Input2):
    img_url = client.DeviceManagerService.TakeImage(Input(serial, path))[0].ImagePath
    # Process image i.e. get the Confluency
    # Returns the confluency parameter and a path to confluency image
    msg = client.ImageProcessingService.GetConfluency(Input2(img_url))
    Confluency = msg[0].Confluency
    ImagePath = msg[0].ImagePath
    return Confluency, ImagePath

def connect_decapper():
    decapper = minimalmodbus.Instrument('COM3', 1)
    decapper.serial.baudrate = 115200 
    print(decapper)
    minimalmodbus.MODE_RTU= 'rtu'
    decapper.write_register(256, 1) #initialize gripper
    decapper.write_register(263, 50) #set rotation speed (range: 1-100)
    return decapper

def run_thermowraps(port='COM1', baudrate=57600):
    with Serial(port='COM1', baudrate=57600) as ser:
        ser.write(b'RUN\r')
        time.sleep(1)
        res = ser.read_all()
        print(res)

def stop_thermowraps(port='COM1', baudrate=57600):
    with Serial(port='COM1', baudrate=57600) as ser:
        ser.write(b'STOP\r')
        time.sleep(1)
        res = ser.read_all()
        print(res)

def change_temperature_thermowraps(temperature, port='COM1', baudrate=57600):
    with Serial(port='COM1', baudrate=57600) as ser:
        command = b'SETTEMP '+str(temperature)+'\r'
        ser.write(command)
        time.sleep(1)
        res = ser.read_all()
        print(res)

def take_flask_out_of_incubator(n_flask, rtde_c, rtde_r, rtde_io_set, gripper):

    """
    This function takes a flask (labeled n_flask) currently in the incubator and put it out

    Initial Position : Robot Waiting Position
    
    Final Position : Robot Waiting Position
    """
    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

    # Unlock Door
    rtde_io_set.setStandardDigitalOut(0, False)
    rtde_io_set.setStandardDigitalOut(4, False)
    rtde_io_set.setStandardDigitalOut(5, True)
    time.sleep(0.2)

    # Open Door
    rtde_io_set.setStandardDigitalOut(1, True)
    time.sleep(3)

    # Move in front of Incubator
    rtde_c.moveJ(j_front_incubator, 2, 2)

    # Open Gripper 
    gripper.open()
    gripper.set_force(20) # from 0 to 100 %
    gripper.set_speed(20) # from 0 to 100 %
    gripper.move(40)

    # Move to Specific Flask Position
    # Flask number 0 top 1 mid 2 bottom
    rtde_c.moveL(pos_front_flask[n_flask], 1, 1)

    # Move to Flask Gripping Position
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_gripping_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-0.190, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_gripping_flask, 0.5, 0.5)

    # Close Gripper
    gripper.close()
        
    # Move upwards
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_upwards = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.008, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_upwards, 0.5, 0.5)

    # Move out of Incubator
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_out = [cur_TCP_pos[0], cur_TCP_pos[1]+0.190, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_out, 0.5, 0.5)

    # Move to Waiting Position for Closing Door
    rtde_c.moveJ(j_waiting_door, 1, 1)

    # Close Door
    rtde_io_set.setStandardDigitalOut(5, False)
    rtde_io_set.setStandardDigitalOut(1, False)
    rtde_io_set.setStandardDigitalOut(0, True)
    time.sleep(3)

    # Lock Door 
    rtde_io_set.setStandardDigitalOut(4, True)

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 1, 1)

    print('Flask '+str(n_flask)+' successfully taken out of incubator')

def place_flask_in_incubator(n_flask, rtde_c, rtde_r, rtde_io_set, gripper):
    
    """
    This function takes the flask currently held by the gripper and place it back in the incubator to position n_flask

    Initial Position : Robot Waiting Position
    
    Final Position : Robot Waiting Position
    """
    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 1, 1)

    # Move to Waiting Position for Opening Door
    rtde_c.moveJ(j_waiting_door, 1, 1)

    # Unlock Door
    rtde_io_set.setStandardDigitalOut(0, False)
    rtde_io_set.setStandardDigitalOut(4, False)
    rtde_io_set.setStandardDigitalOut(5, True)
    time.sleep(0.2)

    # Open Door
    rtde_io_set.setStandardDigitalOut(1, True)
    time.sleep(3)

    # Move in front of Incubator
    rtde_c.moveJ(j_front_incubator, 1, 1)

    # Move to Specific Flask Release Position
    rtde_c.moveL(pos_front_empty[n_flask], 0.5, 0.5)

    # Move to Flask Release Position
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_release_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-0.190, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_release_flask, 0.5, 0.5)

    # Move Down to Flask Rest Position
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_release_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.005, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_release_flask, 0.5, 0.5)
        
    # Release Flask
    gripper.move(44)

    # Move out of Incubator
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_out = [cur_TCP_pos[0], cur_TCP_pos[1]+0.190, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_out, 0.5, 0.5)

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 1, 1)

    # Close Door
    rtde_io_set.setStandardDigitalOut(5, False)
    rtde_io_set.setStandardDigitalOut(1, False)
    rtde_io_set.setStandardDigitalOut(0, True)
    time.sleep(3)

    # Lock Door
    rtde_io_set.setStandardDigitalOut(4, True)

    print('Flask '+str(n_flask)+' successfully placed back in incubator')

def analyze_cells(client, serial, Input, Input2, rtde_c, rtde_r, gripper):
    
    """
    This function takes the flask currently held by the gripper and place it under the microscope to take 6 pictures of cells and decide whether or not cells are ready to be splitted or that media should be changed

    Initial Position : Robot Waiting Position

    Final Position : Robot Waiting Position
    """

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 1, 1)

    # Move close to Regripping Station
    rtde_c.moveJ(j_close2regripping, 1, 1)

    # Move above Regripping Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_touch_regripping = [cur_TCP_pos[0]-0.103, cur_TCP_pos[1]+0.022, cur_TCP_pos[2]+0.002, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_touch_regripping, 0.5, 0.5)

    # Move down to Regripping Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_touch_regripping = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.016, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_touch_regripping, 0.2, 0.2)

    # Release Flask
    gripper.open()

    # Regrip left part of the Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_left_regripping = [cur_TCP_pos[0], cur_TCP_pos[1]-0.05, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_left_regripping, 0.5, 0.5)

    # Close Gripper
    gripper.set_force(30)
    gripper.close()

    # Leave Regripping Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_leave_regripping = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.02, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_leave_regripping, 0.5, 0.5)

    # Move away from Regripping Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_away_regripping = [cur_TCP_pos[0]+0.15, cur_TCP_pos[1]-0.02, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_away_regripping, 0.5, 0.5)

    # Move down below close to Microscope
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_close2micro = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.18, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_close2micro, 0.5, 0.5)
    
    # Move under Microscope for 1st Picture
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_under_micro = [cur_TCP_pos[0]-0.15, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_under_micro, 0.2, 0.2)
    
    # Create Directory for Cell Images + Confluency List
    path = create_directory(folder_name = "Cell_Confluency", path = "C:/Users/Operator/Desktop/Cell Images")
    confluency_val = []

    # Put down Flask on Microscope
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_under_micro = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.006, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_under_micro, 0.2, 0.2)

    # Evaluate Confluency of Cells 1st Picture
    Confluency, ImagePath = take_picture(path, client, serial, Input, Input2)
    confluency_val.append(Confluency)

    # Move +1cm in y direction
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_2_micro = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.006, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_2_micro, 0.2, 0.2)

    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_2_micro = [cur_TCP_pos[0], cur_TCP_pos[1]+0.01, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_2_micro, 0.2, 0.2)

    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_under_micro = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.006, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_under_micro, 0.2, 0.2)

    # Evaluate Confluency of Cells 2nd Picture
    Confluency, ImagePath = take_picture(path, client, serial, Input, Input2)
    confluency_val.append(Confluency)

    # Move +1cm in y direction
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_2_micro = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.006, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_2_micro, 0.2, 0.2)

    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_2_micro = [cur_TCP_pos[0], cur_TCP_pos[1]+0.01, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_2_micro, 0.2, 0.2)

    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_under_micro = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.006, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_under_micro, 0.2, 0.2)   

    # Evaluate Confluency of Cells 3rd Picture
    Confluency, ImagePath = take_picture(path, client, serial, Input, Input2)
    confluency_val.append(Confluency) 

    # Move up from Microscope
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_2_micro = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.006, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_2_micro, 0.2, 0.2)

    # Move away from Microscope
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_2_micro = [cur_TCP_pos[0]+0.15, cur_TCP_pos[1]-0.02, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_2_micro, 0.2, 0.2)

    # Move up close to Regripping Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_close2micro = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.18, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_close2micro, 0.5, 0.5)

    # Move to Regripping Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_away_regripping = [cur_TCP_pos[0]-0.15, cur_TCP_pos[1]+0.02, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_away_regripping, 0.5, 0.5)

    # Put on Regripping Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_away_regripping = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.02, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_away_regripping, 0.2, 0.2)

    # Open Gripper
    gripper.open()

    # Regrip central part of the Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_cent_regripping = [cur_TCP_pos[0], cur_TCP_pos[1]+0.05, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_cent_regripping, 0.5, 0.5)

    # Close Gripper
    gripper.set_force(20)
    gripper.close()

    # Leave Regripping Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_leave_regripping = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.02, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_leave_regripping, 0.5, 0.5)

    # Move away from Regripping Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_close_regripping = [cur_TCP_pos[0]+0.15, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_close_regripping, 0.5, 0.5)

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 1, 1)

    return path

def open_cell_flask(rtde_c, rtde_r, gripper):
    
    """
    This function takes the flask currently held by the gripper and place it in the holder to open it for cell splitting / changing media

    Initial Position : Robot Waiting Position
    
    Final Position : Robot Waiting Position
    """

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 1, 1)

    # Move close to Opening Station
    rtde_c.moveJ(j_open_st, 1, 1)

    # Go down to the Flask Holder slit
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.001, cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Go down while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.046
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]+dy, cur_TCP_pos[2]+dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)

    # Open Gripper
    gripper.open()

    # Go up while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.2
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-dy, cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

    # Move to Waiting Position close to Cell Flask
    rtde_c.moveJ(j_open_w, 2, 2)

    # Move on top of Lid
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Increase Force and Close Gripper
    gripper.set_force(20)
    gripper.close()

    # Decap
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_decap = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5], 0.1, 0.1, 0.0]
    pos_decapping1 = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.015, 0, 3.1415, cur_TCP_pos[5], 0.1, 0.1, 0.005]
    pos_decapping2 = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.03, 2.21, -2.21, cur_TCP_pos[5], 0.1, 0.1, 0.0]
    path_decap = [pos_decap, pos_decapping1, pos_decapping2]
    rtde_c.moveL(path_decap)
    gripper.set_force(20)

    # Move Lid up from Flask
    gripper.move(20)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.02, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Move to Waiting Position close to Cell Flask
    rtde_c.moveJ(j_open_w2, 1, 1)

    # Move to Releasing Lid Joint Configuration
    rtde_c.moveJ(j_release_lid_w, 1, 1) 

    # Move to Lid Position for Cell Flask
    rtde_c.moveJ(j_lid_c, 1, 1)

    # Move above Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]+0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Mode down to Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.01, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.1, 0.1)

    # Release Lid
    gripper.open()

    # Move away of Current Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]-0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Move to Releasing Lid Joint Configuration
    rtde_c.moveJ(j_release_lid_w, 2, 2) 

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

def recap_cell_flask(rtde_c, rtde_r, gripper):
        
    """
    This function recaps the processed cell flask and grip it back

    Initial Position : Robot Waiting Position

    Final Position : Robot Waiting Position
    """

    # Move to Capping Joint Position
    rtde_c.moveJ(j_waiting, 2, 2)

    # Move to Releasing Lid Joint Configuration
    rtde_c.moveJ(j_release_lid_w, 2, 2)

    # Move to Lid Position for Cell Flask
    rtde_c.moveJ(j_lid_c, 2, 2)

    # Move close to Lid
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.014, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]+0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.1, 0.1)

    # Grip Lid
    gripper.set_force(10)
    gripper.close()
    gripper.move(20)

    # Mode up from Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.012, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Move away from Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]-0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)
    rtde_c.moveJ(j_release_lid_w, 1, 1) 
    
    # Move to Waiting Position close to Cell Flask
    rtde_c.moveJ(j_recap_c, 1, 1)

    # Move on top of Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0]-0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Move Lid to Recap Start Position
    gripper.close()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.01, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.1, 0.1)
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.008, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.1, 0.1)
    gripper.close()
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.005, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.1, 0.1)
    gripper.close()

    # Recap
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_recap = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5], 0.1, 0.1, 0.0]
    pos_recapping1 = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.005, 3.1415, 0, cur_TCP_pos[5], 0.1, 0.1, 0.02]
    pos_recapping2 = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.01, 2.21, -2.21, cur_TCP_pos[5], 0.1, 0.1, 0.0]
    path_recap = [pos_recap, pos_recapping1, pos_recapping2]
    rtde_c.moveL(path_recap)

    # Go up from Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.005, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.1, 0.1)
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.1, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)
    gripper.set_force(20) 

    # Move to Capping Joint Configuration
    rtde_c.moveJ(j_recap_c, 2, 2)

    # Move to Robot Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

    # Move close to Open Station
    rtde_c.moveJ(j_close2open, 2, 2)

    # Grab Cell Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    gripper.close()

    # Go up while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.042
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-dy, cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move to Robot Waiting Position
    rtde_c.moveJ(j_waiting, 1, 1)

def add_trypsin(rtde_c, rtde_r, gripper):
    
    """
    This function takes the cell flask emptied of media and washing solution, place it in the holder under the trypsin, add trypsin by pull/push device and return it to pouring station

    Initial Position : Robot Waiting Position
    
    Final Position : Robot Waiting Position
    """

    # Move to Robot Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

    # Move close to Open Station
    rtde_c.moveJ(j_close2open, 2, 2)

    # Grab Cell Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    gripper.close()

    # Go up while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.04
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-dy, cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)

    # Go to Trypsin Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]+0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Go down while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]+dy, cur_TCP_pos[2]+dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.05, 0.05)
    gripper.open()

    # Leave Trypsin Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]+0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Go near Push/Pull Trypsin Device
    rtde_c.moveJ(j_waiting, 2, 2)
    rtde_c.moveJ(j_trypsin, 2, 2)

    # Go close to Trypsin Device
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Pump Trypsin
    gripper.close()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.0075, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)

    # Push Trypsin
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]+0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.1, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.16, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.033, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.05, 0.05)

    # Leave Trypsin Device
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.05, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]+0.16, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move to Trypsin Station
    rtde_c.moveJ(j_trypsin_st, 2 , 2)

    # Grab Cell Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    gripper.close()

    # Leave Trypsin Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.04
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-dy, cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.05, 0.05)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.05, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Put Flask horizontally
    cur_J_pos = rtde_r.getActualQ()
    j_flask_hor = [cur_J_pos[0], cur_J_pos[1], cur_J_pos[2], cur_J_pos[3], cur_J_pos[4], -1.5499709288226526]
    rtde_c.moveJ(j_flask_hor, 1, 1)

    # Shake gently the Flask
    shake(rtde_c=rtde_c, rtde_r=rtde_r)

    # Go to Trypsin Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)

    # Put Flask with 14.5° angle
    cur_J_pos = rtde_r.getActualQ()
    j_flask_hor = [cur_J_pos[0], cur_J_pos[1], cur_J_pos[2], cur_J_pos[3], cur_J_pos[4], -2.848912541066305]
    rtde_c.moveJ(j_flask_hor, 1, 1)

    # Go down while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.05, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.04
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]+dy, cur_TCP_pos[2]+dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.05, 0.05)
    gripper.open()

    # Go up while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.2
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-dy, cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

def trash(rtde_c, rtde_r, gripper, n_thermo = 1): 

    """
    This function takes the cell flask with consumed media / used washing solution and trash it
    If n_thermo = 0, meaning that washing solution just has been washed, the flask is shaken before being trashed

    Initial Position : Robot Waiting Position
    
    Final Position : Robot Waiting Position
    """

    # Move to Robot Waiting Position
    rtde_c.moveJ(j_waiting, 1, 1)

    # Move close to Open Station
    rtde_c.moveJ(j_close2open, 1, 1)

    # Grab Cell Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    gripper.close()

    # Go up while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.042
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-dy, cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.03, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Put Flask horizontally
    cur_J_pos = rtde_r.getActualQ()
    j_flask_hor = [cur_J_pos[0], cur_J_pos[1], cur_J_pos[2], cur_J_pos[3], cur_J_pos[4], -1.5499709288226526]
    rtde_c.moveJ(j_flask_hor, 1, 1)

    if n_thermo == 0 :
        # Shake gently the Flask
        shake(rtde_c=rtde_c, rtde_r=rtde_r)

    # Put Flask horizontally
    cur_J_pos = rtde_r.getActualQ()
    j_flask_hor = [cur_J_pos[0], cur_J_pos[1], cur_J_pos[2], cur_J_pos[3], cur_J_pos[4], -1.5499709288226526]
    rtde_c.moveJ(j_flask_hor, 1, 1)

    # Go to Trashing Position
    rtde_c.moveJ(j_trash, 1, 1)
    
    # Add Pouring Function
    start_pos = rtde_r.getActualTCPPose()
    pour_flask(start_pos=start_pos, stop_angle=75, stop_time=1800, rtde_c=rtde_c, vel_L=0.05, acc_L=1.5, blending=0.001)

    # Go back to Open Station
    rtde_c.moveJ(j_trash_back, 1, 1)

    # Change Flask angle to match Slit angle
    cur_J_pos = rtde_r.getActualQ()
    j_flask_hor = [cur_J_pos[0], cur_J_pos[1], cur_J_pos[2], cur_J_pos[3], cur_J_pos[4], -2.8422098795520228]
    rtde_c.moveJ(j_flask_hor, 1, 1)

    # Go down while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.03, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.04
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]+dy, cur_TCP_pos[2]+dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)
    gripper.open()

    # Leave Open Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.2
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-dy, cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move to Robot Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2) 

def shake(rtde_c, rtde_r):
    # Shake gently the Flask
    cur_J_pos = rtde_r.getActualQ()
    j_flask_shake = [cur_J_pos[0], cur_J_pos[1], cur_J_pos[2], cur_J_pos[3], cur_J_pos[4], cur_J_pos[5]-3*np.pi/180]
    rtde_c.moveJ(j_flask_shake, 0.5, 0.5)
    j_flask_shake = [cur_J_pos[0], cur_J_pos[1], cur_J_pos[2], cur_J_pos[3], cur_J_pos[4], cur_J_pos[5]+3*np.pi/180]
    rtde_c.moveJ(j_flask_shake, 0.5, 0.5)
    j_flask_shake = [cur_J_pos[0], cur_J_pos[1], cur_J_pos[2], cur_J_pos[3], cur_J_pos[4], cur_J_pos[5]-3*np.pi/180]
    rtde_c.moveJ(j_flask_shake, 0.5, 0.5)
    j_flask_shake = [cur_J_pos[0], cur_J_pos[1], cur_J_pos[2], cur_J_pos[3], cur_J_pos[4], cur_J_pos[5]+3*np.pi/180]
    rtde_c.moveJ(j_flask_shake, 0.5, 0.5)

def take_decap_bottle(n_thermo, rtde_c, rtde_r, gripper, decapper):

    """
    This function takes the bottle from the Thermowraps station, decaps it and waits for the pouring
    n_thermo : label of the Thermowraps station
    0 = washing solution
    1 = media
    Initial Position : Robot Waiting Position
    
    Final Position : Pouring Waiting Position
    """

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

    # Get Media Bottle at Thermowraps Station
    rtde_c.moveJ(j_mid_bottle, 2, 2)
    rtde_c.moveJ(j_bottle[n_thermo], 2, 2)

    # Grab Media Bottle by the Top
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.15, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 1, 1)
    gripper.close()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.15, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 1, 1)

    # Go to Middle Position to avoid collision
    rtde_c.moveJ(j_mid_bottle, 2, 2)

    # Go to Regripping Media Station
    rtde_c.moveJ(j_regrip_bottle, 2, 2)

    # Go down to Release Media Bottle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.26, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)

    # Go up to change Joint Configuration
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.15, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)    

    # Change Joint Configuration to Decap Media Bottle
    rtde_c.moveJ(j_change_grip, 2, 2)

    # Go down to pick up Media Bottle
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)

    # Pick up Media Bottle by the Side
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1]+0.025, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)
    gripper.close()

    # Go up to Decapper
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)

    # Set Decapper Parameters
    decapper.write_register(261, 0) #set position 0°
    decapper.write_register(263, 50) #set rotation speed (range: 1-100)

    # Move to Decapper
    rtde_c.moveJ(j_below_decap, 1, 1)
    
    # Go up to Decapper Grip Position
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.06, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)
    decapper.write_register(259, 220) #grip bottle

    # Go down while decapping 
    thread1 = threading.Thread(target=decapper.write_register(261, 1080)) #rotate gripper 1080 degrees
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_decap_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.03, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    thread2 = threading.Thread(target=rtde_c.moveL(pos_decap_media, 0.01, 0.01))
    thread1.start()
    thread2.start()

    # Move away from Decapper
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.06, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)


    # Move to Regripping Station for Photo
    rtde_c.moveJ(j_change_grip, 1, 1)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_media = [cur_TCP_pos[0], cur_TCP_pos[1]+0.025, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_media, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)
    gripper.open()

    return n_thermo

def recap_place_bottle_back(n_thermo, rtde_c, rtde_r, gripper, decapper):

    """
    This function takes the media bottle after pouring, recaps it and places it back in the Thermowraps cooling system
    n_thermo = 0 washing solution
    n_thermo = 1 DPBS
    Initial Position : Change Grip Position
    
    Final Position : Robot Waiting Position
    """
    
    # Move to Pouring Waiting Position
    rtde_c.moveJ(j_change_grip, 1, 1)

    # Move to Decapper
    rtde_c.moveJ(j_below_decap, 1, 1)

    # Go up to Decapper Grip Position
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.05, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)

    # Go up while recapping
    thread1 = threading.Thread(target=decapper.write_register(261, 0)) #rotate gripper of -1080 degrees
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_decap_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.01, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    thread2 = threading.Thread(target=rtde_c.moveL(pos_decap_media, 0.02, 0.02))
    thread1.start()
    thread2.start()

    # Release Media Bottle
    decapper.write_register(259, 1000) #release bottle

    # Go away from Decapper
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.06, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)

    # Move back to Regripping Station
    rtde_c.moveJ(j_change_grip, 1, 1)

    # Pick up Media Bottle by the Side
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_media = [cur_TCP_pos[0], cur_TCP_pos[1]+0.025, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_media, 0.5, 0.5)

    # Go down to Release Media Bottle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)

    # Get away from Regripping Station
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1]-0.05, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)

    # Go up to Change Joint Configuration
    rtde_c.moveJ(j_change_grip, 2, 2)
    rtde_c.moveJ(j_regrip_bottle, 2, 2)

    # Go down to Grip again Media Bottle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.26, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)
    gripper.close()

    # Go up to Middle Position to avoid collision
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.26, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)

    # Go to Thermowraps Station
    rtde_c.moveJ(j_mid_bottle, 2, 2)
    rtde_c.moveJ(j_bottle[n_thermo], 2, 2)

    # Go down to Place Media Bottle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_media = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.15, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_media, 0.5, 0.5)
    gripper.open()

    # Go up to Waiting Position
    rtde_c.moveJ(j_mid_bottle, 2, 2)
    rtde_c.moveJ(j_waiting, 2, 2)

def put_empty_flask_to_filling_station(n_empty, n_holder, rtde_c, rtde_r, gripper):

    """
    This function takes the empty flask labeled n_empty and place it in the holder specified (labeled as n_holder)

    Initial Position : Above Holders

    Final Position : Above Holders
    """

    # Beginning : Take empty Flask to Filling Stations

    # Move to Waiting Position
    rtde_c.moveJ(j_above_holders, 2, 2)

    # Move Front of Empty Flask Station
    rtde_c.moveJ(j_front_empty, 2, 2)

    # Move to Specific Empty Flask Position
    rtde_c.moveL(pos_empty_flask[n_empty], 0.5, 0.5)

    # Close a bit Gripper
    gripper.move(40)

    # Pick Empty Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]+0.172, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Close Gripper
    gripper.close()
    
    # Move up of Empty Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.008, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move away from Empty Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.20, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-0.20, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move close to Holder Positions
    rtde_c.moveJ(j_above_holders, 1, 1)

    # Move to Flask Holder Positions n_holder
    rtde_c.moveJ(j_holder[n_holder], 1, 1)

    # Go down to the Flask Holder slit
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Go down while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.042
    dx = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0]+dx, cur_TCP_pos[1], cur_TCP_pos[2]+dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.05, 0.05)

    # Open Gripper
    gripper.open()

    # Go up while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.2
    dx = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0]-dx, cur_TCP_pos[1], cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)

    # Move to Waiting Position above Holders
    rtde_c.moveJ(j_above_holders, 2, 2)

    #End : Take empty Flask to Filling Stations

def decap_and_place_lid(n_decap, n_lid, rtde_c, rtde_r, gripper):
    
    """
    This function decaps the empty flask labeled n_decap and place it in the lid holder specified (labeled as n_lid)

    Initial Position : Decapping Waiting Position

    Final Position : Decapping Waiting Position
    """

    # Move to Decapping Joint Configuration
    rtde_c.moveJ(j_decap_w, 2, 2)

    # Move to Flask Holder Positions n_decap
    rtde_c.moveJ(j_decap[n_decap], 2, 2)

    # Move on top of Lid 
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]+0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Increase Force and Close Gripper
    gripper.set_force(20)
    gripper.close()

    # Decap
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_decap = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5], 0.1, 0.1, 0.0]
    pos_decapping1 = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.015, 2.21, 2.21, cur_TCP_pos[5], 0.1, 0.1, 0.005]
    pos_decapping2 = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.03, 0, 3.1415, cur_TCP_pos[5], 0.1, 0.1, 0.0]
    path_decap = [pos_decap, pos_decapping1, pos_decapping2]
    rtde_c.moveL(path_decap)
    gripper.move(20)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.02, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)
    gripper.set_force(20)

    # Give space to change Joint Configuration
    rtde_c.moveJ(j_decap_w2, 2, 2)

    # Move to Releasing Lid Joint Configuration
    rtde_c.moveJ(j_release_lid_w, 2, 2) 

    # Move to Lid n_lid Positions
    rtde_c.moveJ(j_lid[n_lid], 2, 2)

    # Move above Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]+0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Mode down to Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.01, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.1, 0.1)

    # Release Lid
    gripper.open()  

    # Move away of Current Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]-0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Move to Releasing Lid Joint Configuration
    rtde_c.moveJ(j_release_lid_w, 2, 2) 

    # Move to Waiting Position above Holders
    rtde_c.moveJ(j_decap_w, 2, 2)

def recap_lid(n_rlid, n_recap, rtde_c, rtde_r, gripper):
        
    """
    This function recaps the now passaged flask labeled n_recap with the lid labeled n_rlid

    Initial Position : Capping Waiting Position

    Final Position : Capping Waiting Position
    """

    # Move to Capping Joint Position
    rtde_c.moveJ(j_decap_w, 2, 2)

    # Move to Releasing Lid Joint Configuration
    rtde_c.moveJ(j_release_lid_w, 2, 2)

    # Move to Lid n_rlid Position
    rtde_c.moveJ(j_rlid[n_rlid], 2, 2)

    # Move close to Lid
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]+0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Grip Lid
    gripper.set_force(10)
    gripper.close()
    gripper.move(20)

    # Mode up from Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.012, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.1, 0.1)

    # Move away from Lid Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]-0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)
    rtde_c.moveJ(j_release_lid_w, 2, 2) 
    
    # Move to Capping Joint Configuration
    rtde_c.moveJ(j_decap_w2, 2, 2)

    # Move to Flask Holder Positions n_recap
    rtde_c.moveJ(j_recap[n_recap], 2, 2)

    # Move on top of Holder
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1]+0.1, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)

    # Move Lid to Recap Start Position
    gripper.close()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.01, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.008, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)
    gripper.close()
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.005, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)
    gripper.close()

    # Recap
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_recap = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5], 0.1, 0.1, 0.0]
    pos_recapping1 = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.005, 2.21, 2.21, cur_TCP_pos[5], 0.1, 0.1, 0.01]
    pos_recapping2 = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.01, 3.1415, 0, cur_TCP_pos[5], 0.1, 0.1, 0.0]
    path_recap = [pos_recap, pos_recapping1, pos_recapping2]
    rtde_c.moveL(path_recap)

    # Go up from Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.005, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)
    gripper.open()
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_top_lid = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.05, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_top_lid, 0.5, 0.5)    

    # Move to Capping Joint Configuration
    rtde_c.moveJ(j_decap_w, 2, 2)

def pour_flask(start_pos, stop_angle, stop_time, rtde_c, vel_L = 0.007, acc_L = 1.5, blending = 0.02):
    # stop angle 50, stop time 1800
    # find TCP path
    stop_angle = int(stop_angle)
    stop_time = int(stop_time)

    to_find = "_" + str(stop_angle) + "_" + str(stop_time)

    print("To find: ", to_find)

    # iterate through all folders in folder_path
    folder_path = r"CellFlask_diff"

    # iterate through all folders in folder_path
    for filename in os.listdir(folder_path):
        # if folder contains to_find
        if to_find in filename:
            scene_path = os.path.join(folder_path, filename)
            # Count the total number of rows in the file
            with open(scene_path, 'r') as file:
                num_rows = sum(1 for _ in file)
            # load datapoints and skip first and last 5
            data_points = np.loadtxt(scene_path, delimiter=',', skiprows=1, max_rows=num_rows-2)
            print("Scene path: ", scene_path)
            print(data_points.shape)
            break

    # convert from inches to meters
    data_points[:, 0] = data_points[:, 0] * 0.0254
    data_points[:, 1] = data_points[:, 1] * 0.0254

    # create list of positions
    positions = []
    for i in range(data_points.shape[0]):
        positions.append([-data_points[i,1], -data_points[i,0], 0.0, 0.0, 0.0, data_points[i,2]]) # will move around x, y of tool and rotate around z of tool --> to be updated for different setups

    positions_converted = []
    for i in range(data_points.shape[0]):
        # if none of the data entries is 0
        if not (positions[i][0] == 0 or positions[i][1] == 0 or positions[i][5] == 0):
            positions_converted.append(rm.PoseTrans(start_pos, positions[i])) # transform from tool coordinate system to base coordinate system

    # get position of the first duplicate converted_position that is not position 0
    for i in range(len(positions_converted)):
        if positions_converted[i] == positions_converted[0]:
            continue
        elif positions_converted[i] == positions_converted[i-1] == positions_converted[i-2] == positions_converted[i-3] == positions_converted[i-4]:
            print(i)
            break

    # count the values that are the same as i
    count = 0
    for j in range(i, len(positions_converted)):
        if positions_converted[j] == positions_converted[i]:
            count += 1
        else:
            break

    # split positions_converted into two lists
    positions_converted1 = positions_converted[0:i]
    positions_converted2 = positions_converted[i:]

    rtde_c.moveL(start_pos, 0.5, 0.5)

    blend_i = blending
    blend_3 = 0.0
    path = []
    for i in range(len(positions_converted1)-1):
        path.append([positions_converted1[i][0], positions_converted1[i][1], positions_converted1[i][2], positions_converted1[i][3], positions_converted1[i][4], positions_converted1[i][5], vel_L, acc_L, blend_i])

    path.append([positions_converted1[-1][0], positions_converted1[-1][1], positions_converted1[-1][2], positions_converted1[-1][3], positions_converted1[-1][4], positions_converted1[-1][5], vel_L, acc_L, 0])
    rtde_c.moveL(path)

    time.sleep(count/89)

    path_2 = []
    for i in range(len(positions_converted2)-1):
        path_2.append([positions_converted2[i][0], positions_converted2[i][1], positions_converted2[i][2], positions_converted2[i][3], positions_converted2[i][4], positions_converted2[i][5], vel_L, acc_L, blend_i])

    path_2.append([positions_converted2[-1][0], positions_converted2[-1][1], positions_converted2[-1][2], positions_converted2[-1][3], positions_converted2[-1][4], positions_converted2[-1][5], vel_L, acc_L, blend_3])

    rtde_c.moveL(path_2)

    rtde_c.moveL(start_pos, 0.5, 0.5)

def grab_passaged_flask(n_holder, rtde_c, rtde_r, gripper):

    """
    This function grabs the now passaged flask labeled n_holder

    Initial Position : Above Holders

    Final Position : Above Holders
    """

    # Move close to Holder Positions
    rtde_c.moveJ(j_above_holders, 2, 2)

    # Move to Flask Holder Positions n_holder
    rtde_c.moveJ(j_holder_back[n_holder], 2, 2)

    # Go down to the Flask Holder slit
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Go down while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.05
    dx = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0]+dx, cur_TCP_pos[1], cur_TCP_pos[2]+dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)

    # Grab Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]+0.05, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    gripper.close()

    # Go up while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.04
    dx = -d*np.sin(13.5*np.pi/180)
    dz = -d*np.cos(13.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0]-dx, cur_TCP_pos[1], cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)

    # Go up from Flask Holder slit
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move to Waiting Position above Holders
    rtde_c.moveJ(j_above_holders, 1, 1)

def split_cells(rtde_c, rtde_r, gripper):
    """
    This function grabs and pours the processed cell flask into the 3 empty flask
    
    Initial Position : Robot Waiting Position

    Final Position : Robot Waiting Position
    """

    # Move to Robot Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

    # Move close to Open Station
    rtde_c.moveJ(j_close2open, 2, 2)

    # Grab Cell Flask
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.1, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    gripper.close()

    # Go up while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.042
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-dy, cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]+0.05, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Passaging
    rtde_c.moveJ(j_neutral, 0.5, 0.5)
    rtde_c.moveJ(j_pour_start_1, 0.5, 0.5)
    start_pos =  rtde_r.getActualTCPPose()
    pour_flask(start_pos, stop_angle=14, stop_time=200, rtde_c=rtde_c, vel_L = 0.015, acc_L = 1.5, blending = 0.001)
    rtde_c.moveJ(j_pour_start_2, 0.5, 0.5)
    start_pos =  rtde_r.getActualTCPPose()
    pour_flask(start_pos, stop_angle=18, stop_time=200, rtde_c=rtde_c, vel_L = 0.015, acc_L = 1.5, blending = 0.001)
    rtde_c.moveJ(j_pour_start_3, 0.5, 0.5)
    start_pos =  rtde_r.getActualTCPPose()
    pour_flask(start_pos, stop_angle=50, stop_time=200, rtde_c=rtde_c, vel_L = 0.015, acc_L = 1.5, blending = 0.001)

    # Move to Waiting Position
    rtde_c.moveJ(j_above_holders, 1, 1)

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 1, 1)

    # Move close to Opening Station
    rtde_c.moveJ(j_open_st, 1, 1)

    # Go down to the Flask Holder slit
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.001, cur_TCP_pos[1], cur_TCP_pos[2]-0.24, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Go down while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.046
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]+dy, cur_TCP_pos[2]+dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)

    # Open Gripper
    gripper.open()

    # Go up while following 14.5° angle
    cur_TCP_pos = rtde_r.getActualTCPPose()
    d = 0.2
    dy = -d*np.sin(14.5*np.pi/180)
    dz = -d*np.cos(14.5*np.pi/180)
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-dy, cur_TCP_pos[2]-dz, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

def put_cell_flask_empty_station(n_empty, rtde_c, rtde_r, gripper):
    """
    This function grabs and puts the now empty cell in the empty flask station

    Initial Position : Robot Waiting Position

    Final Position : Robot Waiting Position
    """
    # Move to Robot Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

    # Move close to Empty Flask Station
    rtde_c.moveJ(j_above_holders, 2, 2)

    # Move Front of Empty Flask Station
    rtde_c.moveJ(j_front_empty, 2, 2)

    # Move to Specific Empty Flask Position
    rtde_c.moveL(pos_empty_back[n_empty], 0.5, 0.5)

    # Go to Free Empty Flask Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]+0.171, cur_TCP_pos[1]+0.006, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    
    # Move up of Empty Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1], cur_TCP_pos[2]-0.01, cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.1, 0.1)
    gripper.open()

    # Move away from Empty Station
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0]-0.20, cur_TCP_pos[1], cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)
    cur_TCP_pos = rtde_r.getActualTCPPose()
    pos_pick_flask = [cur_TCP_pos[0], cur_TCP_pos[1]-0.20, cur_TCP_pos[2], cur_TCP_pos[3], cur_TCP_pos[4], cur_TCP_pos[5]]
    rtde_c.moveL(pos_pick_flask, 0.5, 0.5)

    # Move close to Holder Positions
    rtde_c.moveJ(j_above_holders, 2, 2)

    # Move to Waiting Position
    rtde_c.moveJ(j_waiting, 2, 2)

def pour_bottle(stop_angle, stop_time, rtde_c, vel_L = 0.0074, acc_L = 1.5, blending = 0.0015):
    # find TCP path
    stop_angle = int(stop_angle)
    stop_time = int(stop_time)

    to_find = "_" + str(stop_angle) + "_" + str(stop_time)

    # iterate through all folders in folder_path
    folder_path = r"MediaBottle_diff"

    # iterate through all folders in folder_path
    for filename in os.listdir(folder_path):
        # if folder contains to_find
        if to_find in filename:
            scene_path = os.path.join(folder_path, filename)
            # Count the total number of rows in the file
            with open(scene_path, 'r') as file:
                num_rows = sum(1 for _ in file)
            # load datapoints and skip first and last 5
            data_points = np.loadtxt(scene_path, delimiter=',', skiprows=1, max_rows=num_rows-2)
            print("Scene path: ", scene_path)
            print(data_points.shape)
            break

    # convert from inches to meters
    data_points[:, 0] = data_points[:, 0] * 0.0254
    data_points[:, 1] = data_points[:, 1] * 0.0254
    data_points[:, 2] = data_points[:, 2] + 0.25307274

    # create list of positions
    positions = []
    for i in range(data_points.shape[0]):
        positions.append([data_points[i,0], -data_points[i,1], 0.0, 0.0, 0.0, data_points[i,2]]) # will move around x, y of tool and rotate around z of tool --> to be updated for different setups

    positions_converted = []
    for i in range(data_points.shape[0]):
        # if none of the data entries is 0
        if not (positions[i][0] == 0 or positions[i][1] == 0 or positions[i][5] == 0):
            positions_converted.append(rm.PoseTrans(start_pos_horizontal, positions[i])) # transform from tool coordinate system to base coordinate system

    # get position of the first duplicate converted_position that is not position 0
    for i in range(len(positions_converted)):
        if positions_converted[i] == positions_converted[0]:
            continue
        elif positions_converted[i] == positions_converted[i-1] == positions_converted[i-2] == positions_converted[i-3] == positions_converted[i-4]:
            print(i)
            break

    # count the values that are the same as i
    count = 0
    for j in range(i, len(positions_converted)):
        if positions_converted[j] == positions_converted[i]:
            count += 1
        else:
            break

    # split positions_converted into two lists
    positions_converted1 = positions_converted[0:i]
    positions_converted2 = positions_converted[i:]

    rtde_c.moveL(start_pos_bottle_rotated, 0.1, 0.1)

    blend_i = blending
    path = []
    for i in range(len(positions_converted1)-1):
        path.append([positions_converted1[i][0], positions_converted1[i][1], positions_converted1[i][2], positions_converted1[i][3], positions_converted1[i][4], positions_converted1[i][5], vel_L, acc_L, blend_i])

    path.append([positions_converted1[-1][0], positions_converted1[-1][1], positions_converted1[-1][2], positions_converted1[-1][3], positions_converted1[-1][4], positions_converted1[-1][5], vel_L, acc_L, 0])
    rtde_c.moveL(path)

    time.sleep(count/150)

    path_2 = []
    for i in range(len(positions_converted2)-1):
        path_2.append([positions_converted2[i][0], positions_converted2[i][1], positions_converted2[i][2], positions_converted2[i][3], positions_converted2[i][4], positions_converted2[i][5], vel_L, acc_L, blend_i])

    path_2.append([positions_converted2[-1][0], positions_converted2[-1][1], positions_converted2[-1][2], positions_converted2[-1][3], positions_converted2[-1][4], positions_converted2[-1][5], vel_L, acc_L, 0])
    rtde_c.moveL(path_2)

def move_to_pick_up(rtde_c, gripper):
    gripper.open()
    rtde_c.moveJ(j_pick_up_2, 0.5, 0.5)
    rtde_c.moveL(pick_up_pos_4, 0.5, 0.5)

def estimate_volume(rtde_c):
    rtde_c.moveJ(j_pick_up_2, 0.5, 0.5)
    rtde_c.moveJ(j_camera_1, 0.5, 0.5)
    """
    path = capture_image()
    pred_vol_1 = predict_with_vol(path, save_segmentation=False, save_depth=False, predict_volume=True, vessel_volume=755, no_GPU=True)
    """
    rtde_c.moveJ(j_camera_2, 0.5, 0.5)
    """
    path = capture_image()
    pred_vol_2 = predict_with_vol(path, save_segmentation=False, save_depth=False, predict_volume=True, vessel_volume=755, no_GPU=True)
    """
    #input_start_vol = (pred_vol_1 + pred_vol_2)/2
    input_start_vol = 100
    return input_start_vol

def find_best_parameters(input_start_vol, input_target_vol):
    input_target_vol -= 10 # correction factor
    """
    summary_flask = pd.read_csv('../pouring_simulation/output/summary_medium_final.csv', skiprows=[0,1])
    min_loss = 100000

    for j in range(len(summary_flask)):
        loss = abs(summary_flask.iloc[j,5] - input_start_vol) + abs(summary_flask.iloc[j,7] - input_target_vol) #+ summary_flask.iloc[j,8]*0.3
        if loss < min_loss:
            min_loss = loss
            stop_angle = (summary_flask.iloc[j,3])
            stop_time = (summary_flask.iloc[j,4])*1000
    """
    stop_angle = 50
    stop_time = 1800
    return stop_angle, stop_time

def get_bottle(rtde_c, gripper):
    rtde_c.moveJ(j_pick_up_2, 0.5, 0.5)
    gripper.open()
    rtde_c.moveL(pick_up_pos_4, 0.5, 0.5)
    gripper.close()
    rtde_c.moveL(move_to_pouring_pos_1, 0.3, 0.3)
    rtde_c.moveL(move_to_pouring_pos_2, 0.3, 0.3)
    rtde_c.moveL(move_to_pouring_pos_3, 0.3, 0.3)
    rtde_c.moveL(start_pos_bottle_rotated, 0.3, 0.3)

def place_bottle_back(rtde_c):
    rtde_c.moveL(start_pos_bottle_rotated, 0.3, 0.3)
    rtde_c.moveL(move_to_pouring_pos_3, 0.3, 0.3)
    rtde_c.moveL(move_to_pouring_pos_2, 0.3, 0.3)
    rtde_c.moveJ(j_change_grip, 1, 1)

def autonomous_pouring(rtde_c, gripper, target_vol):
    input_start_vol = estimate_volume(rtde_c)
    stop_angle, stop_time = find_best_parameters(input_start_vol, target_vol)
    get_bottle(rtde_c, gripper)
    pour_bottle(stop_angle, stop_time, rtde_c)
    place_bottle_back(rtde_c)