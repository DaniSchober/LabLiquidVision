import argparse
import src.data.make_dataset as make_dataset
import tkinter as tk
from src.models_1_no_vol.predict_full_pipeline import predict_no_vol
from src.models_2_input_vol.predict_full_pipeline import predict_with_vol
from src.models_1_no_vol.test_model import test as test_no_vol
from src.models_1_no_vol.predict_vol import predict as predict_no_vol_from_depth_maps
import src.models_1_no_vol.train_model as train_no_vol
from src.models_2_input_vol.test_model import test as test_with_vol
from src.models_2_input_vol.predict_vol import (
    predict as predict_with_vol_from_depth_maps,
)
import src.models_2_input_vol.train_model as train_with_vol


"""

    Main file, starts training, prediction, testing, data recording or data conversion depending on the arguments
    
"""


def main():
    """

    Main function

    """

    # if input arg is train then train else predict
    parser = argparse.ArgumentParser(description="Volume Estimation")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or predict or test or convert or record or predict_on_depth_maps",
    )
    parser.add_argument(
        "--no_GPU",
        action="store_true",
        default=False,
        help="Do not use GPU for prediction",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../segmentation_and_depth/models/segmentation_and_depth.torch",
        help="Path to model for converting the dataset",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="example/image.png",
        help="Path to image for prediction",
    )

    parser.add_argument(
        "--path_input",
        type=str,
        default="data/interim",
        help="Path to input dataset for conversion",
    )

    parser.add_argument(
        "--path_output",
        type=str,
        default="data/processed",
        help="Path to output dataset for conversion",
    )

    parser.add_argument(
        "--folder_path",
        type=str,
        default="data/processed/",
        help="Path to folder containing the converted LabLiquidVolume dataset",
    )

    parser.add_argument(
        "--vessel_volume",
        type=int,
        default=0,
        help="Volume of vessel",
    )

    parser.add_argument(
        "--use_vessel_volume",
        action="store_true",
        default=False,
        help="Use vessel volume as input for model",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train for",
    )

    args = parser.parse_args()

    if args.use_vessel_volume == False:
        print("Using model without vessel volume.")

        if args.mode == "train":
            print("Training model.")
            train_no_vol.run_training(
                data_dir=args.folder_path, num_epochs=args.num_epochs
            )

        elif args.mode == "predict":
            print("Predicting volume of liquid in image.")
            predict_no_vol(
                args.image_path,
                predict_volume=True,
                save_segmentation=False,
                save_depth=False,
                no_GPU=args.no_GPU,
            )

        elif args.mode == "test":
            print("Testing model.")
            test_no_vol(data_dir=args.folder_path)

        elif args.mode == "predict_on_depth_maps":
            print("Predicting volume of liquid based on depth maps.")
            predict_no_vol_from_depth_maps(folder_path=args.folder_path)

        elif args.mode == "convert":
            print("Converting dataset.")
            print("This might take a while...")
            make_dataset.create_converted_dataset(
                path_input=args.path_input,
                path_output=args.path_output,
                model_path=args.model_path,
            )

        elif args.mode == "record":
            from src.data.record_data import App

            print("Recording dataset.")
            if __name__ == "__main__":
                root = tk.Tk()
                app = App(root)
                root.mainloop()

        else:
            print("Invalid argument!")
            print("Please enter train or predict or convert or record.")

    elif args.use_vessel_volume == True:
        print("Using model with vessel volume.")
        # if train then train else predict
        if args.mode == "train":
            print("Training model.")
            train_with_vol.run_training(
                data_dir=args.folder_path, num_epochs=args.num_epochs
            )

        elif args.mode == "predict":
            print("Predicting volume of liquid in vessel.")
            predict_with_vol(
                args.image_path,
                predict_volume=True,
                save_segmentation=False,
                save_depth=False,
                vessel_volume=args.vessel_volume,
                no_GPU=args.no_GPU,
                show_visualization=True,
            )

        elif args.mode == "test":
            print("Testing model.")
            test_with_vol(data_dir=args.folder_path)

        elif args.mode == "predict_on_depth_maps":
            print("Predicting volume of liquid based on depth maps.")
            predict_with_vol_from_depth_maps(folder_path=args.folder_path)

        else:
            print("Invalid argument!")
            print(
                "Please enter train or predict or test or do not use --use_vessel_volume."
            )


if __name__ == "__main__":
    main()
