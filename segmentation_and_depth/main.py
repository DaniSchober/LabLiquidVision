import argparse
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import src.models.predict_model as predict
import src.models.train_model as train_model
from src.models.evaluate_model_new import evaluate

"""
This is the main file to run the segmentation and depth prediction of the project

It can be run in three modes:
    --mode train: train the model
    --mode predict: predict the segmentation and depth of an image
    --mode evaluate: evaluate the model on a folder of images

The main function takes in the following arguments:
    --mode: train or predict or evaluate
    --cuda: True or False
    --model_path: path to trained model
    --batch_size: batch size for training
    --epochs: number of epochs for training
    --load_model: True or False (loading of pretrained model)
    --use_labpics: True or False (use lab pictures for training)
    --image_path: path to image to predict
    --folder_path: path to folder for evaluation

Example usage:
    python main.py --mode train --cuda True --batch_size 6 --epochs 75 --load_model False --use_labpics True

    python main.py --mode predict --cuda True --model_path models/segmentation_depth_model.torch --image_path example/RGBImage10.png

    python main.py --mode evaluate --cuda True --model_path models/segmentation_depth_model.torch --folder_path data/interim/TranProteus1/Testing/LiquidContent

"""


def main():
    print("Starting main function")
    # if input arg is train then train else predict
    parser = argparse.ArgumentParser(description="Segmentation and Depth Estimation")
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
        default="models/segmentation_depth_model.torch",
        help="Path to model",
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
        default=75,
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
        "--image_path",
        type=str,
        default="example/RGBImage.png",
        help="Path to image",
    )

    parser.add_argument(
        "--folder_path",
        type=str,
        default="data/interim/TranProteus1/Testing/LiquidContent",
        help="Folder path for evaluation",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # start training
        print("Training model")
        train_model.train(
            args.batch_size, args.epochs, args.load_model, args.use_labpics
        )

    elif args.mode == "predict":
        # start predicting
        print("Predicting model")
        predict.predict(model_path=args.model_path, image_path=args.image_path)

    elif args.mode == "evaluate":
        # start evaluating
        print("Evaluating model")
        evaluate(model_path=args.model_path, folder_path=args.folder_path)

    else:
        print("Invalid argument")
        print("Please enter train or predict")


if __name__ == "__main__":
    main()
