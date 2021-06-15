import argparse
import sys

import matplotlib.pyplot as plt
import torch

from src.models.model import MyAwesomeModel

plt.style.use("seaborn-dark")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction arguments")
    parser.add_argument("--load_model_from", default="models/checkpoint.pth")
    parser.add_argument("--image_folder", default=None)
    parser.add_argument("--pickle_file", default=None)
    args = parser.parse_args(sys.argv[2:])

    # if args.image_folder is not None:
    # Load images
    # TODO: Add loop to loads and transforms images
    images = torch.rand(10, 28, 28)

    # Load model
    state_dict = torch.load(args.load_model_from)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    model.eval()

    # Make predictions
    outputs = model(images)
    ps = torch.exp(outputs)
    predictions = ps.max(1)[1]
    print(list(predictions.numpy()))
