import torch
import pytest

from src.models.model import MyAwesomeModel

@pytest.mark.parametrize("num_filter", [1, 10, 15])
def test_output_size(num_filter):
    model = MyAwesomeModel(num_filter=num_filter)
    output = model(torch.randn(1, 1, 28, 28))
    assert list(output.shape) == [1, 10]
