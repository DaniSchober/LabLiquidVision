# main file, starts training or prediction depeeending on the arguments
import argparse
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import src.models.predict_model as predict
import src.models.train_model as train_model
import src.models.train_model_epochs as train_model_epochs


def main():
    print("Starting main function")
    # if input arg is train then train else predict
    parser = argparse.ArgumentParser(description="Depth Estimation")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or predict",
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
        default="models/55__03042023-2211.torch",
        help="Path to model",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="example/RGBImage10.png",
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

    args = parser.parse_args()

    # print args with "args"
    print("Args: " + str(args))

    # if train then train else predict
    if args.mode == "train":
        print("Training model")
        train_model_epochs.train(
            args.batch_size, args.epochs, args.load_model, args.use_labpics
        )

    elif args.mode == "predict":
        # start src.models.predict_model.py
        print("Predicting model")
        predict.predict(model_path=args.model_path, image_path=args.image_path)

    else:
        print("Invalid argument")
        print("Please enter train or predict")

    # end main function
    # return args


if __name__ == "__main__":
    main()
