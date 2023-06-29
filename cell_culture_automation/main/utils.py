'''
Function definitions for the main script.

'''

import time
import rtde_control
import rtde_receive
from rtde_control import Path, PathEntry
import rtde_io
import os

def connect_robot(ip = "192.168.2.1"):
    rtde_c = rtde_control.RTDEControlInterface(ip) #IP address found on robot
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    rtde_io_set = rtde_io.RTDEIOInterface(ip)
    return rtde_c, rtde_r, rtde_io_set

def pour_flask(start_pos, stop_angle, stop_time, rtde_c, vel_L = 0.007, acc_L = 1.5, blending = 0.015):
    # find TCP path
    stop_angle = int(stop_angle)
    stop_time = int(stop_time)

    to_find = "_" + str(stop_angle) + "_" + str(stop_time)

    print("To find: ", to_find)

    # iterate through all folders in folder_path
    folder_path = r"../pouring_simulation/output/CellFlask_diff"

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

    rtde_c.stopScript()
            

