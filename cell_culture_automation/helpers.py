import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta
from time import sleep
import rtde_control
import rtde_receive
from rtde_control import Path, PathEntry
import rtde_io
import copy
import math
import random
from random import randint


"""
Define important positions
"""
home_pos = [
    -0.19161405312765759,
    0.13472399156690068,
    0.365305811889139,
    -2.216389383069919,
    -2.214478541184546,
    -0.011795250688296986,
]


def connect_robot(ip="192.168.2.1"):
    rtde_c = rtde_control.RTDEControlInterface(ip)  # IP address found on robot
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    rtde_io_set = rtde_io.RTDEIOInterface(ip)
    return rtde_c, rtde_r, rtde_io_set


def move_above_goal(home_pos, middle_pos, rtde_c, vel_J=3, acc_J=3):
    path = Path()
    blend = 0.1
    path.addEntry(
        PathEntry(
            PathEntry.MoveJ,
            PathEntry.PositionTcpPose,
            [
                middle_pos[0],
                middle_pos[1],
                middle_pos[2],
                middle_pos[3],
                middle_pos[4],
                middle_pos[5],
                vel_J,
                acc_J,
                blend,
            ],
        )
    )
    rtde_c.movePath(path, False)
    return 0
