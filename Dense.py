import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 activation: str = None, name: str = None, use_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.activation_name = activation.lower() if isinstance(activation, str) else None
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.activation_name in ("none", None):
            return x
        elif self.activation_name == "relu":
            return F.relu(x)
        elif self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation_name == "tanh":
            return torch.tanh(x)
        elif self.activation_name == "softmax":
            return F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
