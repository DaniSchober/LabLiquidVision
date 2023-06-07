import pyrealsense2 as rs
import cv2
from datetime import datetime
import os
import numpy as np
import sys
import warnings


def capture_image():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the pipeline
    pipeline.start(config)

    # Wait for a coherent color frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert the color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Generate a timestamp for the image filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a directory for saving the captured images
    save_dir = "captured_images/" + str(timestamp) + "/"
    os.makedirs(save_dir, exist_ok=True)

    # Save the color image
    image_path = os.path.join(save_dir, f"image.png")
    cv2.imwrite(image_path, color_image)
    print("Image saved")

    # Stop the pipeline
    pipeline.stop()

    return image_path
