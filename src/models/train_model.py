import argparse
import sys

import joblib
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as transforms
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, MLFlowLogger
from src.models.model import MNIST_CNN
from src.util import Classifier, MNISTData
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self, single_step=False):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if single_step:
            args.command = "train"

        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        if single_step:
            self.weights = getattr(self, args.command)(single_step=True)
        else:
            getattr(self, args.command)()

    def train(self, single_step=False):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=3e-4, type=float)
        parser.add_argument("--num_epochs", default=10, type=int)
        parser.add_argument("--num_filters", default=16, type=int)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # Initializations
        model = MyAwesomeModel(num_filter=args.num_filters)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Load data
        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=64, shuffle=True
        )

        # Prepare metrics
        train_losses = []
        steps = 0
        print_every = 200
        running_loss = 0
        epoch_losses = []
        epoch_loss = 0

        # Initialize logger
        if not single_step:
            wandb.init(config=vars(args))
        else:
            wandb.init(config=vars(args), mode="disabled")
            init_weights = model.conv1.weight.clone().detach()

        wandb.watch(model, log_freq=print_every)

        for epoch in range(args.num_epochs):
            batch_losses = []
            for images, labels in trainloader:
                # Increment counter and zero gradients
                steps += 1

                # Forward pass
                logits = model(images.view(-1, 1, 28, 28))
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Step
                optimizer.step()

                # Save losses
                batch_losses += [loss.item()]
                train_losses += [loss.item()]
                running_loss += loss.item()
                epoch_loss += loss.item()

                if single_step:
                    return init_weights, model.conv1.weight

                # Print loss
                if steps % print_every == 0:
                    wandb.log({"loss": loss})
                    running_loss = 0

            epoch_losses += [epoch_loss]
            wandb.log({"epoch_loss": epoch_loss})
            epoch_loss = 0

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--load_model_from", default="models/checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        if args.load_model_from:
            state_dict = torch.load(args.load_model_from)
            model = MyAwesomeModel()
            model.load_state_dict(state_dict)
            model.eval()

        _, test_set = mnist()
        criterion = nn.NLLLoss()
        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        accuracy = 0
        test_loss = 0

        for images, labels in testloader:
            output = model(images.view(-1, 1, 28, 28))
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = labels.data == ps.max(1)[1]
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        print(f"Accuracy on test set: {accuracy / len(testloader):.4f}")


def parse_arguments():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--num_filters", default=16, type=int)
    parser.add_argument("--hidden_size1", default=128, type=int)
    parser.add_argument("--hidden_size2", default=128, type=int)
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--dropout", default=0.25, type=float)
    parser.add_argument("-azure", action='store_true')
    # add any additional argument that you want
    args = parser.parse_args()
    if args.azure:
        print('Running on Azure.')
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    # Setup data, model, logger
    mnist = MNISTData(batch_size=64)
    img_model = MNIST_CNN(
        hidden_size1=args.hidden_size1,
        hidden_size2=args.hidden_size2,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )

    model = Classifier(model=img_model, lr=args.lr)

    print('making prediction...')
    img = torch.rand(10, 1, 28, 28)
    img = img.tolist()
    input_json = json.dumps({"data": img})
    headers = {'Content-Type': 'application/json'}
    data = np.array(json.loads(input_json)['data'])
    mtp = model.predict_step(data, None)
    print('prediction:')
    print(mtp)
    assert getattr(model, 'infer', None) != None

    logger = WandbLogger(project="MNIST", log_model="all", config=args)

    # Setup early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.05, patience=5, verbose=True, mode="max"
    )

    # Setup checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="models/",
        filename="mnist-{epoch:02d}-{val_acc:.2f}",
        mode="max",
    )

    # Setup trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.num_epochs,
        precision=32,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Fit model
    trainer.fit(model, mnist)

    # Test models
    result = trainer.test()

    if args.azure:
        model_file = 'outputs/mnist_model.pkl'
        os.makedirs('outputs', exist_ok=True)
        joblib.dump(value=model, filename=model_file)
        print(f'Saving model under {model_file}.')

    # Terminate logger
    wandb.finish()
