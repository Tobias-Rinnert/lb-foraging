from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import Adam

NEURON_PROB = 0.3   # per-layer neuron mutation probability
LAYER_PROB = 0.1    # topology change probability
NEURON_MIN = 2
NEURON_MAX = 64


def build_nn(input_size: int, hidden_layers: list[int]) -> nn.Sequential:
    """
    Build a feedforward network: Linear→ReLU per hidden layer, then Linear→Sigmoid.

    build_nn(11, [5]) produces the same shape as the old hardcoded architecture.
    """
    layers: list[nn.Module] = []
    prev = input_size
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def mutate_architecture(
    hidden_layers: list[int],
    neuron_prob: float = NEURON_PROB,
    layer_prob: float = LAYER_PROB,
) -> list[int]:
    """
    Return a mutated copy of hidden_layers (never mutates in-place).

    - Per layer: with neuron_prob, apply ±(1..3) delta, clamped to [NEURON_MIN, NEURON_MAX]
    - With layer_prob: 50/50 add or remove a layer (min 1 layer always kept)
    """
    result = list(hidden_layers)

    # per-layer neuron mutation
    for i in range(len(result)):
        if random.random() < neuron_prob:
            delta = random.randint(1, 3) * random.choice([-1, 1])
            result[i] = max(NEURON_MIN, min(NEURON_MAX, result[i] + delta))

    # topology mutation
    if random.random() < layer_prob:
        if random.random() < 0.5 or len(result) == 1:
            # add a layer — size = mean of neighbours, clamped
            if len(result) == 1:
                new_size = max(NEURON_MIN, min(NEURON_MAX, result[0]))
            else:
                idx = random.randint(0, len(result) - 1)
                neighbours = []
                if idx > 0:
                    neighbours.append(result[idx - 1])
                if idx < len(result) - 1:
                    neighbours.append(result[idx + 1])
                new_size = max(NEURON_MIN, min(NEURON_MAX, int(sum(neighbours) / len(neighbours))))
                result.insert(idx, new_size)
                return result
            result.append(new_size)
        else:
            # remove a layer (min 1 kept)
            if len(result) > 1:
                idx = random.randint(0, len(result) - 1)
                result.pop(idx)

    return result


def transfer_weights(src: nn.Sequential, dst: nn.Sequential) -> None:
    """
    Copy overlapping weights layer-by-layer from src into dst.

    For each matching Linear layer pair: copies [:min_out, :min_in] weights
    and [:min_out] bias. Extra neurons in dst keep their random init.
    Silently skips non-Linear layers.
    """
    src_linears = [m for m in src.modules() if isinstance(m, nn.Linear)]
    dst_linears = [m for m in dst.modules() if isinstance(m, nn.Linear)]

    with torch.no_grad():
        for s, d in zip(src_linears, dst_linears):
            min_out = min(s.out_features, d.out_features)
            min_in = min(s.in_features, d.in_features)
            d.weight[:min_out, :min_in] = s.weight[:min_out, :min_in]
            d.bias[:min_out] = s.bias[:min_out]


@dataclass
class AgentGenome:
    agent_id: int
    hidden_layers: list[int]
    fitness: float
    nn_model: nn.Sequential
    optimizer: Adam
    _extra: dict = field(default_factory=dict, repr=False)


def reproduce(
    parent_genomes: list[AgentGenome],
    food_eaten_counts: dict[int, int],
    foods_per_child: int,
    input_size: int,
) -> list[AgentGenome]:
    """
    Produce children from surviving parents based on food eaten.

    Each parent gets floor(food_eaten / foods_per_child) children.
    Each child: mutate_architecture once, build_nn, transfer_weights from parent, fresh Adam.
    Children receive sequential agent_ids starting at 0.
    Returns empty list if no one produced any children.
    """
    children: list[AgentGenome] = []
    next_id = 0

    for parent in parent_genomes:
        n_children = food_eaten_counts.get(parent.agent_id, 0) // max(1, foods_per_child)
        for _ in range(n_children):
            new_arch = mutate_architecture(parent.hidden_layers)
            new_model = build_nn(input_size, new_arch)
            transfer_weights(parent.nn_model, new_model)
            new_opt = Adam(new_model.parameters())
            children.append(
                AgentGenome(
                    agent_id=next_id,
                    hidden_layers=new_arch,
                    fitness=0.0,
                    nn_model=new_model,
                    optimizer=new_opt,
                )
            )
            next_id += 1

    return children
