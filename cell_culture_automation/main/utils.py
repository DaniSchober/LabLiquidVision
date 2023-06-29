'''
Function definitions for the main script.

'''

def connect_robot(ip = "192.168.2.1"):
    rtde_c = rtde_control.RTDEControlInterface(ip) #IP address found on robot
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    rtde_io_set = rtde_io.RTDEIOInterface(ip)
    return rtde_c, rtde_r, rtde_io_set

    