import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.datasets import MNIST


def mnist():
    # Convert to tensors and normalise

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Download and load the train data
    trainset = datasets.MNIST(
        "data/processed/MNIST_data/", download=True, train=True, transform=transform
    )

    # Download and load the test data
    testset = datasets.MNIST(
        "data/processed/MNIST_data/", download=True, train=False, transform=transform
    )

    return trainset, testset


class Classifier(pl.LightningModule):
    def __init__(self, model, lr: float = 3e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.val_accuracy = torchmetrics.Accuracy(num_classes=10)
        self.train_accuracy = torchmetrics.Accuracy(num_classes=10)
        self.test_accuracy = torchmetrics.Accuracy(num_classes=10)
        self.save_hyperparameters()

    def training_step(self, batch, batch_id):
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        self.train_accuracy(F.softmax(logits, dim=-1), labels)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        metric = torchmetrics.Accuracy()
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        self.val_accuracy(F.softmax(logits, dim=-1), labels)
        self.log("val_acc", self.val_accuracy, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        self.test_accuracy(F.softmax(logits, dim=-1), labels)
        self.log("test_acc", self.test_accuracy)
        self.log("test_loss", loss)
        return self.test_accuracy

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def predict_step(self, batch, batch_idx = None):
        x = torch.from_numpy(batch)
        x = x.type(torch.FloatTensor)
        outputs = self.model(x)
        ps = F.softmax(outputs, dim=-1)
        ps = ps.max(1)[1]
        ps = ps.numpy()
        return ps

class MNISTData(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.data_dir = "data/processed/MNIST_data/"
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.batch_size = batch_size

    def setup(self, stage=None):
        mnist_full = MNIST(
            self.data_dir, train=True, download=True, transform=self.transform
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(
            self.data_dir, train=False, download=True, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)
