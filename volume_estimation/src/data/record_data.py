import pyrealsense2 as rs
import numpy as np
import os
import cv2
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Vessel Capture")
        self.master.geometry("400x300")  # set the window size to 400x300 pixels
        self.pipeline = None
        self.config = None
        self.device_product_line = None

        self.root = tk.Frame(self.master)
        self.root.pack()

        tk.Label(self.root, text="Vessel Name:").grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )
        tk.Label(self.root, text="Vessel Volume (ml):").grid(
            row=1, column=0, padx=10, pady=10, sticky="w"
        )
        tk.Label(self.root, text="Liquid Volume (ml):").grid(
            row=2, column=0, padx=10, pady=10, sticky="w"
        )

        self.vessel_name_entry = tk.Entry(self.root)
        self.vessel_name_entry.grid(row=0, column=1, padx=10, pady=10)

        self.vessel_vol_entry = tk.Entry(self.root)
        self.vessel_vol_entry.grid(row=1, column=1, padx=10, pady=10)

        self.liquid_vol_entry = tk.Entry(self.root)
        self.liquid_vol_entry.grid(row=2, column=1, padx=10, pady=10)

        self.capture_button = tk.Button(self.root, text="Capture", command=self.capture)
        self.capture_button.grid(row=3, column=0, padx=10, pady=10)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.quit)
        self.quit_button.grid(row=3, column=1, padx=10, pady=10)

        self.root.mainloop()

    def capture(self):
        if not self.pipeline:
            self.init_pipeline()

        vessel_name = self.vessel_name_entry.get()
        vol_vessel = self.vessel_vol_entry.get()
        vol_liquid = self.liquid_vol_entry.get()

        if not vessel_name or not vol_liquid or not vol_vessel:
            messagebox.showerror("Error", "Please fill in all the fields.")
            return

        try:
            vol_liquid = int(vol_liquid)
            vol_vessel = int(vol_vessel)
        except ValueError:
            messagebox.showerror(
                "Error", "Please enter a valid integer value for the volumes."
            )
            return

        today = datetime.now()

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            color_image = cv2.resize(
                color_image,
                dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                interpolation=cv2.INTER_AREA,
            )

        # Create directory for saving the captured data
        path = (
            f"data/interim/{vessel_name}_{vol_liquid}ml_{today.strftime('%d%m_%M%S')}"
        )
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        # Save images
        cv2.imwrite(f"{path}/Input_RGBImage.png", color_image)
        cv2.imwrite(f"{path}/Input_DepthMap.png", depth_colormap)

        np.save(path + "/Input_RGBImage.npy", color_image)
        np.save(path + "/Input_DepthMap.npy", depth_image)

        with open(path + "/Input_vol_liquid.txt", "w") as f:
            f.write(str(vol_liquid))
        with open(path + "/Input_vessel.txt", "w") as f:
            f.write(vessel_name)
        with open(path + "/Input_vol_vessel.txt", "w") as f:
            f.write(str(vol_vessel))

        self.liquid_vol_entry.delete(
            0, "end"
        )  # delete volume entry for easier data entry

        messagebox.showinfo(
            "Capture Done", "Images have been captured and saved successfully!"
        )
        # show image in window
        plt.figure(figsize=(5, 5))

    def init_pipeline(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline_profile = self.pipeline.start(self.config)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

    def quit(self):
        # self.pipeline.stop()
        self.root.quit()


# if __name__ == '__main__':
#    root = tk.Tk()
#   app = App(root)
#   root.mainloop()
