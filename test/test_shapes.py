from src.model import CovidResNet
import torch

def test_forward_shape():
    m = CovidResNet()
    out = m(torch.randn(2,3,224,224))
    assert out.shape == (2,2)
