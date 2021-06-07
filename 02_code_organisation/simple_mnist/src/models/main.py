import sys
import argparse

import numpy as np
import torch
from torch import nn
from model import MyAwesomeModel
import matplotlib.pyplot as plt
from util import mnist
from torchvision import datasets
plt.style.use('seaborn-dark')

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--num_epochs', default=10, type=int)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        train_losses = []
        steps = 0
        print_every = 200
        running_loss = 0
        epoch_losses = []
        epoch_loss = 0

        for epoch in range(args.num_epochs):
            for images, labels in trainloader:
                # Increment counter and zero gradients
                steps += 1
                optimizer.zero_grad()

                # Forward pass
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                # Save and print losses
                train_losses += [loss.item()]
                running_loss += loss.item()
                epoch_loss += loss.item()
                if steps % print_every == 0:
                    print(f'[Epoch {epoch}/{args.num_epochs}]:')
                    print(f'Training loss: {running_loss/print_every:.3f}')
                    running_loss = 0

            epoch_losses += [epoch_loss]
            epoch_loss = 0

        # Save final model
        torch.save(model.state_dict(), 'models/checkpoint.pth')

        # Plot resulting training losses
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(np.arange(1, steps+1), train_losses, label='Training losses (per batch)')
        ax[1].plot(np.arange(1, len(epoch_losses)+1), epoch_losses, label='Training losses (per epoch)',
                   color="#F58A00")
        ax[0].legend()
        ax[1].legend()
        plt.tight_layout()
        plt.savefig('reports/figures/01_final_exercise_training_losses.png')


    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="models/checkpoint.pth")
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
            output = model(images)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        print(f"Accuracy on test set: {accuracy/len(testloader):.4f}")

if __name__ == '__main__':
    TrainOREvaluate()

