"""Neural network architecture builders for fixed-capacity row-based inputs.

Provides build_nn() to create configurable feedforward networks and add_agent_rows()
to extend networks with new agent rows using warm-start weight copying.
"""

import copy
import random
import torch
import torch.nn as nn


def build_nn(input_size: int, hidden_layers: list[int]) -> nn.Sequential:
    """Build a feedforward NN with configurable hidden layers, ending in Linear(1) + Sigmoid.

    Args:
        input_size: number of input features
        hidden_layers: list of hidden layer widths, e.g. [5] or [8, 4]

    Returns:
        nn.Sequential model
    """
    layers = []
    prev = input_size
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def add_agent_rows(model: nn.Sequential, n_new: int) -> nn.Sequential:
    """Extend the first Linear layer by n_new agent rows (2 columns each).

    New columns copy weights from a randomly chosen existing agent row
    (columns 1..end, skipping the global fruit-level neuron at col 0).
    This is a warm start — new rows inherit plausible initial weights.

    Args:
        model: the existing nn.Sequential (must have a Linear first layer with
               in_features == 1 + K*2 for some K >= 1)
        n_new: number of new agent rows to append

    Returns:
        a new nn.Sequential with in_features increased by 2*n_new
    """
    model = copy.deepcopy(model)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    first = linears[0]
    K_old = (first.in_features - 1) // 2

    with torch.no_grad():
        extra_cols = []
        for _ in range(n_new):
            donor = random.randint(0, K_old - 1)
            col = 1 + donor * 2
            extra_cols.append(first.weight[:, col:col + 2].clone())
        new_weight = torch.cat([first.weight] + extra_cols, dim=1)
        new_linear = nn.Linear(first.in_features + 2 * n_new, first.out_features)
        new_linear.weight = nn.Parameter(new_weight)
        new_linear.bias = nn.Parameter(first.bias.clone())

    new_layers = []
    replaced = False
    for layer in model:
        if isinstance(layer, nn.Linear) and not replaced:
            new_layers.append(new_linear)
            replaced = True
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)
