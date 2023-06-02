import pyrealsense2 as rs
import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import threading
from datetime import datetime
import sys
import warnings

# add parent folder to sys.path
sys.path.insert(0, os.path.abspath(".."))

# ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ignore RuntimeWarning messages
warnings.filterwarnings("ignore", category=RuntimeWarning)

from volume_estimation.src.models_no_input_vol.predict_full_pipeline import predict

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Vessel Capture")
        self.master.geometry("400x300")  # set the window size to 400x300 pixels
        self.pipeline = None
        self.window_name = "RGB Stream - Press q to quit"

        self.root = tk.Frame(self.master)
        self.root.pack()

        self.capture_button = tk.Button(self.root, text="Capture", command=self.capture)
        self.capture_button.grid(row=0, column=0, padx=10, pady=10)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.quit)
        self.quit_button.grid(row=0, column=1, padx=10, pady=10)

        t = threading.Thread(target=self.capture_frames)
        t.start()

        self.root.mainloop()

    def capture(self):
        if not self.pipeline:
            self.init_pipeline()

        today = datetime.now()

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Create directory for saving the captured data
        path = f"test_images/{today.strftime('%d%m_%M%S')}"
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        # Save image
        print(color_image.shape)
        cv2.imwrite(f"{path}/Input_RGBImage.png", color_image)
        self.image_path = f"{path}/Input_RGBImage.png"
        np.save(path + "/Input_RGBImage.npy", color_image)


        print("Image has been captured and saved successfully! Path:", path)

        # Predict volume
        self.predict(self.image_path)

    def init_pipeline(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline_profile = self.pipeline.start(self.config)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

    def capture_frames(self):
        if not self.pipeline:
            self.init_pipeline()
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow("RGB Stream - Press q to quit", color_image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break


    def quit(self):
        cv2.destroyAllWindows()
        self.pipeline.stop()
        self.root.quit()
        self.root.destroy()

    def predict(self, image_path):
        predict(image_path)
        messagebox.showinfo("Prediction", "Prediction has been made successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()