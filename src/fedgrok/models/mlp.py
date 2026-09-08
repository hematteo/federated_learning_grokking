"""Generic ReLU MLP for the Omnigrok MNIST setup.

A plain multi-layer perceptron with an adjustable initialisation scale. The
large-init trick is load-bearing for Omnigrok grokking: with standard init the
network just learns MNIST and there is no delay; scaling the initial weights up
(init_scale > 1) places the weight norm outside the "Goldilocks zone" so it has
to drift inward, which is what produces the delayed generalisation.

Biases are included (unlike GrokNet) — this is an ordinary classifier MLP, not
the analytically-special mean-field network. It exposes `P` for the metrics
interface but no `W1`, and `dft_applicable` requires both, so the GrokNet Fourier
metrics correctly skip it.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int, output_dim: int,
                 n_layers: int = 3, init_scale: float = 1.0):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.P = output_dim              # mirror GrokNet.P for the metrics interface

        dims = [input_dim] + [hidden_width] * (n_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

        # Omnigrok large init: scale every parameter up by init_scale.
        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(init_scale)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)        # logits (no activation on the output)
