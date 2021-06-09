from src.models.util import mnist
import torch

def test_number_of_datapoints():
    train, test = mnist()
    assert len(train) == 60000
    assert len(test) == 10000

def test_dimensions():
    train, test = mnist()
    assert list(train.data.shape)[1:] == [28, 28]
    assert list(test.data.shape)[1:] == [28, 28]

def test_all_labels():
    train, test = mnist()
    train_labels = torch.unique(train.targets)
    test_labels = torch.unique(test.targets)
    labels = torch.arange(10, dtype=float)
    assert train_labels in labels
    assert test_labels in labels
