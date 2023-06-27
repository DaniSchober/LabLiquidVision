# main file, starts training or prediction depeeending on the arguments
import argparse
import src.data.make_dataset as make_dataset  # uncommented for without torch
import tkinter as tk
from src.models_no_vol.predict_full_pipeline import predict_no_vol
from src.models_input_vol.predict_full_pipeline import predict_with_vol


def main():
    print("Starting main function")
    # if input arg is train then train else predict
    parser = argparse.ArgumentParser(description="Volume Estimation")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or predict or test or convert or record",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Use cuda",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../segmentation_and_depth/models/55__03042023-2211.torch",
        help="Path to model",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="Path to image for prediction",
    )

    parser.add_argument(
        "--path_input",
        type=str,
        default="data/interim",
        help="Path to input dataset",
    )

    parser.add_argument(
        "--path_output",
        type=str,
        default="data/processed",
        help="Path to output dataset",
    )

    parser.add_argument(
        "--liquid_depth",
        type=str,
        default="data/processed/Tube_50mL_34ml_2404_5523/Input_ContentDepth_segmented.npy",
        help="Depth of liquid in vessel (.npy)",
    )

    parser.add_argument(
        "--vessel_depth",
        type=str,
        default="data/processed/Tube_50mL_34ml_2404_5523/Input_EmptyVessel_Depth_segmented.npy",
        help="Depth of vessel (.npy)",
    )

    parser.add_argument(
        "--folder_path",
        type=str,
        # default="data/processed/Tube_50mL_34ml_2404_5523",
        default="data/processed/Pyrex_100mL_84ml_2404_5319",
        help="Path to folder containing the .npy files",
    )

    parser.add_argument(
        "--input_vol",
        type=bool,
        default=False,
        help="Use input volume",
    )

    parser.add_argument(
        "--vessel_volume",
        type=int,
        default=0,
        help="Volume of vessel",
    )


    args = parser.parse_args()

    if args.input_vol == False:
        # if train then train else predict
        if args.mode == "train":
            print("Training model")
            import src.models_no_vol.train_model

        elif args.mode == "predict":
            # start src.models.predict_model.py
            print("Predicting model")
            predict_no_vol(args.image_path, predict_volume=True, save_segmentation=False, save_depth=False)

        elif args.mode == "test":
            print("Testing model")
            import src.models_no_vol.test_model

        elif args.mode == "convert":
            print("Converting dataset")
            print("This might take a while...")
            make_dataset.create_converted_dataset(
                path_input=args.path_input,
                path_output=args.path_output,
                model_path=args.model_path,
            )

        elif args.mode == "record":
            # start src.data.make_dataset.py
            from src.data.record_data import App

            print("Recording dataset")
            if __name__ == "__main__":
                root = tk.Tk()
                app = App(root)
                root.mainloop()

        else:
            print("Invalid argument")
            print("Please enter train or predict or convert or record")

    elif args.input_vol == True:
        # if train then train else predict
        if args.mode == "train":
            print("Training model")
            import src.models_input_vol.train_model

        elif args.mode == "predict":
            # start src.models.predict_model.py
            print("Predicting volume of liquid in vessel")
            predict_with_vol(args.image_path, predict_volume=True, save_segmentation=False, save_depth=False, vessel_volume=args.vessel_volume)

        elif args.mode == "test":
            print("Testing model")
            import src.models_input_vol.test_model

        else:
            print("Invalid argument")
            print("Please enter train or predict or test or change input_vol to False")


if __name__ == "__main__":
    main()
