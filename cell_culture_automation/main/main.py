# Import UI modules
import tkinter as tk
from tkinter import messagebox
from tkinter.simpledialog import askinteger
from PIL import Image, ImageTk
from tkinter import ttk
from glob import glob
import os

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

# Import Robot positions
from positions import *

# Import Functions
from utils import *

def connect_device():
    global rtde_c, rtde_r, rtde_io_set, gripper, client, serial, Input, Input2, decapper
    rtde_c, rtde_r, rtde_io_set, gripper = connect_robot()
    client, serial, Input, Input2 = connect_microscope()
    decapper = connect_decapper()
    messagebox.showinfo("Connection", "Devices Connected")

def analyze_cell_growth():
    i = askinteger('Which flask ?', 'Enter flask number in incubator (from top to bottom):')
    if i not in range(1,4):
        messagebox.showinfo('Error', "Wrong User Input")
    else:
        messagebox.showinfo("Currently Running", "Analyze Cell Growth of flask n°"+str(i))
    take_flask_out_of_incubator(i-1, rtde_c, rtde_r, rtde_io_set, gripper)
    path = analyze_cells(client, serial, Input, Input2, rtde_c, rtde_r, gripper)
    # iterate through all images in the folder
    j = 0
    for filename in os.listdir(path):
        if filename.endswith("_0.jpg"):
            file_path = os.path.join(path, filename)
            image = Image.open(file_path)
            image = image.resize((440, 440), Image.LANCZOS)
            image_tk = ImageTk.PhotoImage(image)
            label = tk.Label(image_window[j], image=image_tk)
            label.pack()
            j+=1
    place_flask_in_incubator(i-1, rtde_c, rtde_r, rtde_io_set, gripper)
        
def change_media():
    i = askinteger('Which flask ?', 'Enter flask number in incubator (from top to bottom):')
    if i not in range(1,4):
        messagebox.showinfo('Error', "Wrong User Input")
    else:
        messagebox.showinfo("Currently Running", "Changing Media of flask n°"+str(i))
    take_flask_out_of_incubator(i-1, rtde_c, rtde_r, rtde_io_set, gripper)
    path = analyze_cells(client, serial, Input, Input2, rtde_c, rtde_r, gripper)
    # iterate through all images in the folder
    j = 0
    for filename in os.listdir(path):
        if filename.endswith("_0.jpg"):
            file_path = os.path.join(path, filename)
            image = Image.open(file_path)
            image = image.resize((440, 440), Image.LANCZOS)
            image_tk = ImageTk.PhotoImage(image)
            label = tk.Label(image_window[j], image=image_tk)
            label.pack()
            j+=1
    open_cell_flask(rtde_c, rtde_r, gripper)
    trash(rtde_c, rtde_r, gripper)
    n_thermo = take_decap_bottle(0, rtde_c, rtde_r, gripper, decapper)
    target_vol = 30
    autonomous_pouring(rtde_c, gripper, target_vol)
    recap_place_bottle_back(0, rtde_c, rtde_r, gripper, decapper)
    trash(rtde_c, rtde_r, gripper, n_thermo=n_thermo)
    n_thermo = take_decap_bottle(1, rtde_c, rtde_r, gripper, decapper)
    target_vol = 60
    autonomous_pouring(rtde_c, gripper, target_vol)
    recap_place_bottle_back(1, rtde_c, rtde_r, gripper, decapper)
    recap_cell_flask(rtde_c, rtde_r, gripper)
    place_flask_in_incubator(i-1, rtde_c, rtde_r, rtde_io_set, gripper)

def passage():
    i = askinteger('Which flask ?', 'Enter flask number in incubator (from top to bottom):')
    if i not in range(1,4):
        messagebox.showinfo('Error', "Wrong User Input")
    else:
        messagebox.showinfo("Currently Running", "Passassing flask n°"+str(i))
    take_flask_out_of_incubator(i-1, rtde_c, rtde_r, rtde_io_set, gripper)
    path = analyze_cells(client, serial, Input, Input2, rtde_c, rtde_r, gripper)
    # iterate through all images in the folder
    j = 0
    for filename in os.listdir(path):
        if filename.endswith("_0.jpg"):
            file_path = os.path.join(path, filename)
            image = Image.open(file_path)
            image = image.resize((440, 440), Image.LANCZOS)
            image_tk = ImageTk.PhotoImage(image)
            label = tk.Label(image_window[j], image=image_tk)
            label.pack()
            j+=1
    open_cell_flask(rtde_c, rtde_r, gripper)
    trash(rtde_c, rtde_r, gripper)
    n_thermo = take_decap_bottle(0, rtde_c, rtde_r, gripper, decapper)
    target_vol = 30
    autonomous_pouring(rtde_c, gripper, target_vol)
    recap_place_bottle_back(0, rtde_c, rtde_r, gripper, decapper)
    trash(rtde_c, rtde_r, gripper, n_thermo=n_thermo)
    add_trypsin(rtde_c, rtde_r, gripper)
    n_thermo = take_decap_bottle(1, rtde_c, rtde_r, gripper, decapper)
    target_vol = 60
    autonomous_pouring(rtde_c, gripper, target_vol)
    recap_place_bottle_back(1, rtde_c, rtde_r, gripper, decapper)
    put_empty_flask_to_filling_station(0, 0, rtde_c, rtde_r, gripper)
    put_empty_flask_to_filling_station(1, 1, rtde_c, rtde_r, gripper)
    put_empty_flask_to_filling_station(2, 2, rtde_c, rtde_r, gripper)
    decap_and_place_lid(0, 0, rtde_c, rtde_r, gripper)
    decap_and_place_lid(1, 1, rtde_c, rtde_r, gripper)
    decap_and_place_lid(2, 2, rtde_c, rtde_r, gripper)
    split_cells(rtde_c, rtde_r, gripper)
    recap_lid(0, 0, rtde_c, rtde_r, gripper)
    recap_lid(1, 1, rtde_c, rtde_r, gripper)
    recap_lid(2, 2, rtde_c, rtde_r, gripper)
    grab_passaged_flask(0, rtde_c, rtde_r, gripper)
    place_flask_in_incubator(0, rtde_c, rtde_r, rtde_io_set, gripper)
    grab_passaged_flask(1, rtde_c, rtde_r, gripper)
    place_flask_in_incubator(1, rtde_c, rtde_r, rtde_io_set, gripper)
    grab_passaged_flask(2, rtde_c, rtde_r, gripper)
    place_flask_in_incubator(2, rtde_c, rtde_r, rtde_io_set, gripper)
    recap_cell_flask(rtde_c, rtde_r, gripper)
    put_cell_flask_empty_station(0, rtde_c, rtde_r, gripper)

# Create the main window
window = tk.Tk()
window.title("Cell Culture Automation User Interface")
window.geometry("1600x1000")

# Create a themed style
style = ttk.Style()
style.theme_use("alt")

# Customize the window
window.configure(background="#00165E")  # Set the background color of the window
style.configure("TButton",
                background="lightgray",
                foreground="black",
                font=("Arial", 16),
                padding=10,
                width=20,
                relief="raised")
style.configure("TLabel",
                background="#00165E",
                foreground="white",
                font=("Arial", 24, "bold"),
                padding=5)

# Create buttons
button0 = ttk.Button(window, text="Connect Devices", command=connect_device, style="TButton")
button0.place(x=100, y=400)

button1 = ttk.Button(window, text="Analyze Cell Growth", command=analyze_cell_growth, style="TButton")
button1.place(x=100, y=500)

button2 = ttk.Button(window, text="Change Media", command=change_media, style="TButton")
button2.place(x=100, y=600)

button3 = ttk.Button(window, text="Passage", command=passage, style="TButton")
button3.place(x=100, y=700)

# Create a smaller window on the right side
image_window = []
image_window1 = tk.Frame(window, width=440, height=440)
image_window1.place(x=560, y=60)
image_window.append(image_window1)
image_window2 = tk.Frame(window, width=440, height=440)
image_window2.place(x=1020, y=60)
image_window.append(image_window2)
image_window3 = tk.Frame(window, width=440, height=440)
image_window3.place(x=790, y=520)
image_window.append(image_window3)
title_label = ttk.Label(window, text='Cell Confluency', style="TLabel")
title_label.place(x=880, y=0)
window_label = ttk.Label(window, text='Cell Culture Automation\n        User Interface', style="TLabel")
window_label.place(x=50, y=300)
logo = tk.Frame(window, width=225, height=225, bg="#00165E")
logo.place(x=122, y=44)
image = Image.open("logo.png")
image_tk = ImageTk.PhotoImage(image)
label = tk.Label(logo, image=image_tk)
label.pack()

# Start the main event loop
window.mainloop()
