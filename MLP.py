from typing import List
import torch
import torch.nn as nn
from Dense import DenseLayer

def activation_factory(act_type: str):
    if act_type.lower() == 'relu':
        return nn.ReLU()
    elif act_type.lower() == 'tanh':
        return nn.Tanh()
    elif act_type.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {act_type}")


class MLPLayer(nn.Module):
    def __init__(self, input_dim: int, dims: List[int], activate: str = None, name: str = None):
        super().__init__()
        model_dims = [input_dim] + dims

        linears = []
        for i, (i_dim, o_dim) in enumerate(zip(model_dims[:-1], model_dims[1:])):
            if i != len(dims) - 1:
                linears.append(DenseLayer(i_dim, o_dim, activate))
            else:
                linears.append(DenseLayer(i_dim, o_dim))
        self.linears = nn.ModuleList(linears)

    def forward(self, hidden_states):
        for linear in self.linears:
            hidden_states = linear(hidden_states)
        return hidden_states
