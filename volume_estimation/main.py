# main file, starts training or prediction depeeending on the arguments
import argparse
import src.data.make_dataset as make_dataset # uncommented for without torch

import tkinter as tk
from src.data.record_data import App
import src.models.predict_vol as predict_vol


def main():
    print("Starting main function")
    # if input arg is train then train else predict
    parser = argparse.ArgumentParser(description="Volume Estimation")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or predict or convert or record",
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
        help="Path to image",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="Batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of epochs",
    )

    parser.add_argument(
        "--load_model",
        type=bool,
        default=False,
        help="Load model",
    )

    parser.add_argument(
        "--use_labpics",
        type=bool,
        default=True,
        help="Use lab pictures",
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
        #default="data/processed/Tube_50mL_34ml_2404_5523",
        default="data/processed/Pyrex_100mL_84ml_2404_5319",
        help="Path to folder containing the .npy files",
    )

    args = parser.parse_args()

    # print args with "args"
    #print("Args: " + str(args))

    # if train then train else predict
    if args.mode == "train":
        print("Training model")
        import src.models.train_model
        # train_model_epochs.train(args.batch_size, args.epochs, args.load_model, args.use_labpics)

    elif args.mode == "predict":
        # start src.models.predict_model.py
        print("Predicting model")
        predict_vol.predict(args.folder_path)
        # predict_model.predict(model_path=args.model_path, image_path=args.image_path)

    elif args.mode == "convert":
        # start src.data.make_dataset.py
        print("Converting dataset")
        print("This might take a while...")
        make_dataset.create_converted_dataset(
            path_input=args.path_input,
            path_output=args.path_output,
            model_path=args.model_path,
        )

    elif args.mode == "record":
        # start src.data.make_dataset.py
        print("Recording dataset")
        if __name__ == "__main__":
            root = tk.Tk()
            app = App(root)
            root.mainloop()

    else:
        print("Invalid argument")
        print("Please enter train or predict or convert or record")

    # end main function
    # return args


if __name__ == "__main__":
    main()
