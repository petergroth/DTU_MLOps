import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self, num_filter=16):
        super().__init__()
        self.num_classes = 10
        self.skip = False
        self.num_filter = num_filter

        # Layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.num_filter, kernel_size=5
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.num_filter, out_channels=self.num_filter, kernel_size=3
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=self.num_filter * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)

        # Activation and dropout
        self.lrelu = nn.LeakyReLU()
        self.dropout2d = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input to be a 4D Tensor")
        if list(x.shape[1:]) != [1, 28, 28]:
            raise ValueError("Expected images to be of size [1, 28, 28]")

        # Conv1 + activation
        x = self.lrelu(
            self.conv1(x)
        )  # [batch, num_filter, (28-(5-1)-1)+1, (28-(5-1)-1)+1] ~ [batch, num_filter, 24, 24]
        x = self.maxpool(x)  # [batch, 16, 12, 12]
        # Conv2 -> activation
        x = self.lrelu(
            self.conv2(x)
        )  # [batch, 16, (12-(3-1)-1)+1, (12-(3-1)-1)+1] ~ [batch, 16, 10, 10]
        x = self.maxpool(x)  # [batch, 16, 5, 5]
        # Fully connected
        x = x.view(-1, self.num_filter * 5 * 5)
        x = self.dropout(self.lrelu(self.fc1(x)))
        if self.skip:
            x = self.dropout(x + self.lrelu(self.fc2(x)))
        x = self.dropout(self.lrelu(self.fc3(x)))

        return F.log_softmax(x, dim=1)


# FashionMNIST from day 7.
class MNIST_CNN(pl.LightningModule):
    def __init__(self, hidden_size1=600, hidden_size2=128, kernel_size=3, dropout=0.25):
        super().__init__()

        self.num_classes = 10
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=self.kernel_size, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=self.hidden_size1)
        self.drop = nn.Dropout2d(self.dropout)
        self.fc2 = nn.Linear(
            in_features=self.hidden_size1, out_features=self.hidden_size2
        )
        self.fc3 = nn.Linear(
            in_features=self.hidden_size2, out_features=self.num_classes
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
