import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..networks import init_weights, init_seq

class InverseAtlasnet(nn.Module):
    def __init__(self, num_primitives, code_size, primitive_type):
        super().__init__()

        if primitive_type == 'square':
            self.output_dim = 2
        else:
            self.output_dim = 3

        self.input_dim = 3
        self.num_layers = 2
        self.hidden_neurons = 128

        self.linear1 = nn.Linear(self.input_dim, self.hidden_neurons, bias=False)
        init_weights(self.linear1)

        self.linear_list = nn.ModuleList(
            [
                nn.Linear(self.hidden_neurons, self.hidden_neurons, bias=False)
                for i in range(self.num_layers)
            ]
        )

        for l in self.linear_list:
            init_weights(l)

        self.last_linear = nn.Linear(self.hidden_neurons, self.output_dim, bias=False)
        init_weights(self.last_linear)

        self.activation = F.relu

    def forward(self, x):
        """
        Args:
            points: :math:`(N,*,3)`
        """
        x = self.linear1(x)
        x = self.activation(x)
        for i in range(self.num_layers):
            x = self.activation(self.linear_list[i](x))
        x = self.last_linear(x)

        if self.output_dim == 2:
            uv = torch.tanh(x)
        else:
            uv = F.normalize(x, dim=-1)

        return uv
