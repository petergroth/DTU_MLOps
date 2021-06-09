import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.skip = False
        self.num_filter = 16

        # Layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.num_filter, kernel_size=5
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.num_filter, out_channels=self.num_filter, kernel_size=3
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)

        # Activation and dropout
        self.lrelu = nn.LeakyReLU()
        self.dropout2d = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # [batch, 1, H, W]

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
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout(self.lrelu(self.fc1(x)))
        if self.skip:
            x = self.dropout(x + self.lrelu(self.fc2(x)))
        x = self.dropout(self.lrelu(self.fc3(x)))

        return F.log_softmax(x, dim=1)
