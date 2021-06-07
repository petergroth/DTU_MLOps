import sys
import argparse

import numpy as np
import torch
from torch import nn
from data import mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt
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

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        train_losses = []
        steps = 0
        print_every = 200
        running_loss = 0

        for epoch in range(args.num_epochs):
            for images, labels in trainloader:
                steps += 1
                optimizer.zero_grad()

                # Forward pass
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_losses += [loss.item()]
                running_loss += loss.item()
                if steps % print_every == 0:
                    print(f'[Epoch {epoch}/{args.num_epochs}]:')
                    print(f'Training loss: {running_loss/print_every:.3f}')
                    running_loss = 0

        torch.save(model.state_dict(), 'checkpoint_noskip.pth')


        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(np.arange(1,steps+1), train_losses, label='Training losses (per batch)')
        #ax.plot(np.arange(len(all_losses)), all_losses, label='Training losses (per datum)')
        #ax.set_xticks(np.arange(1,args.num_epochs+1))
        plt.legend()
        plt.show()





    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="checkpoint_noskip.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
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

        print(accuracy/len(testloader))

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    