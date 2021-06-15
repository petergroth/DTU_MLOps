
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
import argparse
import sys
from src.models.model import MyAwesomeModel
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.models.model import MNIST_CNN
from src.util import Classifier, MNISTData
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST


def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction arguments')
    parser.add_argument('--model_path', default="models/checkpoint.pth")
    parser.add_argument('--image_folder', default=None)
    parser.add_argument('--pickle_file', default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
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


