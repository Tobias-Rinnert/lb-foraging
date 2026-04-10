import sys
import os
import torch
import torch.nn as nn
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neuroevolution import (
    build_nn, mutate_architecture, transfer_weights, AgentGenome, reproduce,
    NEURON_MIN, NEURON_MAX,
)
from torch.optim import Adam


def _make_genome(agent_id: int, hidden_layers: list[int], fitness: float = 0.0, input_size: int = 11) -> AgentGenome:
    model = build_nn(input_size, hidden_layers)
    return AgentGenome(
        agent_id=agent_id,
        hidden_layers=list(hidden_layers),
        fitness=fitness,
        nn_model=model,
        optimizer=Adam(model.parameters()),
    )


# --- build_nn ---

def test_build_nn_shape():
    model = build_nn(11, [5])
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears[0].in_features == 11
    assert linears[0].out_features == 5
    assert linears[1].in_features == 5
    assert linears[1].out_features == 1


def test_build_nn_output_range():
    model = build_nn(11, [8, 4])
    x = torch.randn(3, 11)
    out = model(x)
    assert out.shape == (3, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_build_nn_multi_hidden():
    model = build_nn(11, [16, 8, 4])
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 4  # 3 hidden + 1 output


# --- mutate_architecture ---

def test_mutate_neuron_bounds():
    layers = [10]
    for _ in range(200):
        result = mutate_architecture(layers, neuron_prob=1.0, layer_prob=0.0)
        for n in result:
            assert NEURON_MIN <= n <= NEURON_MAX


def test_mutate_layer_add_remove():
    layers = [8, 8]
    saw_add = False
    saw_remove = False
    for _ in range(500):
        result = mutate_architecture(layers, neuron_prob=0.0, layer_prob=1.0)
        if len(result) > len(layers):
            saw_add = True
        elif len(result) < len(layers):
            saw_remove = True
    assert saw_add and saw_remove


def test_mutate_single_layer_never_below_one():
    layers = [5]
    for _ in range(200):
        result = mutate_architecture(layers, neuron_prob=0.0, layer_prob=1.0)
        assert len(result) >= 1


def test_mutate_does_not_mutate_in_place():
    original = [10, 10]
    result = mutate_architecture(original, neuron_prob=1.0, layer_prob=0.0)
    assert original == [10, 10]  # unchanged


# --- transfer_weights ---

def test_transfer_weights_same_shape():
    src = build_nn(5, [8])
    dst = build_nn(5, [8])
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(1.0)
    transfer_weights(src, dst)
    for s, d in zip(src.parameters(), dst.parameters()):
        assert torch.allclose(s, d)


def test_transfer_weights_larger_dst():
    # src has 3 hidden neurons, dst has 5 — first 3 should be copied
    src = build_nn(5, [3])
    dst = build_nn(5, [5])
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(2.0)
    transfer_weights(src, dst)
    src_linears = [m for m in src.modules() if isinstance(m, nn.Linear)]
    dst_linears = [m for m in dst.modules() if isinstance(m, nn.Linear)]
    # first 3 rows of dst first linear should equal src
    assert torch.allclose(dst_linears[0].weight[:3], src_linears[0].weight[:3])


def test_transfer_weights_smaller_dst():
    # src has 5 hidden neurons, dst has 3 — dst gets first 3 rows
    src = build_nn(5, [5])
    dst = build_nn(5, [3])
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(3.0)
    transfer_weights(src, dst)
    src_linears = [m for m in src.modules() if isinstance(m, nn.Linear)]
    dst_linears = [m for m in dst.modules() if isinstance(m, nn.Linear)]
    assert torch.allclose(dst_linears[0].weight, src_linears[0].weight[:3])


# --- reproduce ---

def test_reproduce_child_count():
    g = _make_genome(0, [5], input_size=11)
    children = reproduce([g], {0: 6}, foods_per_child=2, input_size=11)
    assert len(children) == 3


def test_reproduce_zero_food():
    g = _make_genome(0, [5], input_size=11)
    children = reproduce([g], {0: 0}, foods_per_child=3, input_size=11)
    assert children == []


def test_reproduce_ids_sequential():
    g0 = _make_genome(0, [5], input_size=11)
    g1 = _make_genome(1, [5], input_size=11)
    # g0 → 2 children, g1 → 1 child
    children = reproduce([g0, g1], {0: 4, 1: 2}, foods_per_child=2, input_size=11)
    ids = [c.agent_id for c in children]
    assert ids == list(range(len(children)))


def test_reproduce_weight_transfer():
    g = _make_genome(0, [5], input_size=11)
    with torch.no_grad():
        for p in g.nn_model.parameters():
            p.fill_(7.0)
    children = reproduce([g], {0: 3}, foods_per_child=1, input_size=11)
    assert len(children) == 3
    # at least some weights should be 7.0 (partial transfer)
    for child in children:
        all_weights = torch.cat([p.flatten() for p in child.nn_model.parameters()])
        assert (all_weights == 7.0).any()
