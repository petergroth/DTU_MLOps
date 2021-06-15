import pytest
import torch

from src.models.main import TrainOREvaluate
from src.models.model import MyAwesomeModel


def test_weight_change():
    init_weights, step_weights = TrainOREvaluate(single_step=True).weights
    assert not torch.all(torch.eq(init_weights, step_weights))


def test_forward_raise():
    with pytest.raises(ValueError):
        model = MyAwesomeModel()
        model.forward(torch.rand(1, 1, 28, 27))
