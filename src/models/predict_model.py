import argparse

import torch
import torch.nn.functional as F

from src.util import Classifier


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prediction arguments")
    parser.add_argument("--model_path", default="models/checkpoint.pth")
    parser.add_argument("--image_folder", default=None)
    parser.add_argument("--pickle_file", default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # Random images
    images = torch.rand(10, 1, 28, 28)

    # Load model
    classifier = Classifier.load_from_checkpoint(args.model_path)
    classifier.eval()

    # Make predictions
    outputs = classifier.model(images)
    ps = F.softmax(outputs, dim=-1)
    predictions = ps.max(1)[1]
    print(list(predictions.numpy()))
