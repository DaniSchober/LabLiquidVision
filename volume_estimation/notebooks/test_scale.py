import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

folder_path = "../data/processed/"
scale_test_folder = "../data/scale_test"


# Function to get user input for distance selection
def get_distance_selection():
    while True:
        selection = input("Select distance (close, medium, far): ").lower()
        if selection in ['close', 'medium', 'far']:
            return selection
        else:
            print("Invalid selection. Please try again.")

# Iterate through all subfolders
for subfolder in os.listdir(scale_test_folder):
    subfolder_path = os.path.join(scale_test_folder, subfolder)

    # Skip if not a directory
    if not os.path.isdir(subfolder_path):
        continue

    # if "Gibco_250" is in subfolder name
    if "Duran_250" in subfolder:
        image_path = os.path.join(subfolder_path, "image.png")
        distance_txt_path = os.path.join(subfolder_path, "distance_selection.txt")

        # Check if image and distance selection file exist
        if not os.path.exists(image_path):
            print(f"Could not find image at {image_path}")
            continue

        if os.path.exists(distance_txt_path):
            print(f"Distance selection already exists for {subfolder}. Skipping.")
            continue

        # Load and display the image
        image = imread(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        # Get distance selection from the user
        distance_selection = get_distance_selection()

        # Save distance selection in a text file
        with open(distance_txt_path, "w") as f:
            f.write(distance_selection)
