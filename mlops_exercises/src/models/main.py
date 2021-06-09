import argparse
import sys

import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn

from models.model import MyAwesomeModel
from models.util import mnist

plt.style.use('seaborn-dark')

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self, single_step=False):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if single_step:
            args.command = 'train'

        if not hasattr(self, args.command):
            print('Unrecognized command')

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        if single_step:
            self.weights = getattr(self, args.command)(single_step=True)
        else:
            getattr(self, args.command)()
    
    def train(self, single_step=False):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--num_epochs', default=10, type=int)
        parser.add_argument('--num_filters', default=16, type=int)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # Initializations
        model = MyAwesomeModel(num_filter=args.num_filters)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Load data
        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

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
            wandb.init(config=vars(args), mode='disabled')
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

            # Plot resulting training losses
            #fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            #ax[0].plot(np.arange(1, steps + 1), train_losses, label='Training losses (per batch)')
            #ax[1].plot(np.arange(1, len(epoch_losses) + 1), epoch_losses, label='Training losses (per epoch)',
            #           color="#F58A00")
            #ax[0].legend()
            #ax[1].legend()
            #wandb.log({"Losses": wandb.Image(plt)})



        # Save final model
        #torch.save(model.state_dict(), 'models/checkpoint.pth')


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
            output = model(images.view(-1, 1, 28, 28))
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        print(f"Accuracy on test set: {accuracy/len(testloader):.4f}")

if __name__ == '__main__':
    TrainOREvaluate()


