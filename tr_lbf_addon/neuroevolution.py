"""Neural network architecture, mutation, and reproduction for agent evolution.

Provides:
- AgentPredictor: attention-based model for predicting agent fruit-targeting probability
- mutate_predictor_dims: randomly shift embedding_dim and decision_hidden hyperparameters
- transfer_predictor_weights: copy overlapping weights from a parent to a child predictor
- AgentGenome: dataclass storing an agent's architecture, model, and fitness
- reproduce: create a new generation of agents from parent genomes
"""

import math
from dataclasses import dataclass, field as dc_field
import numpy as np
import torch
import torch.nn as nn


class AgentPredictor(nn.Module):
    """Attention-based predictor for agent fruit-targeting probability.

    Architecture:
        1. agent_encoder (shared phi): encodes each agent's (level, distance) into
           a fixed-size embedding. Applied with shared weights to focal and all others.
        2. attention: focal embedding queries other embeddings to produce a weighted
           context vector capturing which other agents are most relevant.
        3. decision_net (rho): takes [fruit_level, focal_level, focal_dist, context]
           and outputs a probability in [0, 1].

    The shared encoder ensures the network generalizes across agent slots.
    Attention pooling makes the architecture permutation-invariant over other agents
    while preserving n-th order belief reasoning (all agents visible in one pass).

    Attributes:
        embedding_dim: dimensionality of per-agent embeddings
        agent_encoder: shared network encoding (level, dist) → embedding
        attention_query: projects focal embedding into query space
        decision_net: final network producing target probability
    """

    def __init__(self, embedding_dim: int = 8, decision_hidden: int = 16) -> None:
        """Initialize the predictor.

        Args:
            embedding_dim: size of per-agent embeddings (default 8)
            decision_hidden: hidden layer size in the decision network (default 16)
        """
        super().__init__()
        self.embedding_dim: int = embedding_dim
        "Dimensionality of per-agent embeddings"

        self.agent_encoder: nn.Sequential = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.ReLU(),
        )
        "Shared encoder: (level, distance) → embedding"

        self.attention_query: nn.Linear = nn.Linear(embedding_dim, embedding_dim)
        "Projects focal embedding into query space for attention"

        # decision input: fruit_level (1) + focal_level (1) + focal_dist (1) + context (embedding_dim)
        decision_input_size = 3 + embedding_dim
        self.decision_net: nn.Sequential = nn.Sequential(
            nn.Linear(decision_input_size, decision_hidden),
            nn.ReLU(),
            nn.Linear(decision_hidden, 1),
            nn.Sigmoid(),
        )
        "Final decision network: [fruit, focal, context] → probability"

    def forward(self, fruit_level: torch.Tensor, focal_features: torch.Tensor,
                others_features: torch.Tensor) -> torch.Tensor:
        """Predict probability that the focal agent targets the given fruit.

        Args:
            fruit_level: tensor of shape (batch, 1) — normalized fruit level
            focal_features: tensor of shape (batch, 2) — focal agent [level, distance]
            others_features: tensor of shape (batch, n_others, 2) — other agents [level, distance]

        Returns:
            Tensor of shape (batch, 1) — probability in [0, 1]
        """
        # Encode focal agent
        focal_embedding = self.agent_encoder(focal_features)       # (batch, embedding_dim)

        # Encode all other agents with shared weights
        batch_size = others_features.shape[0]
        n_others = others_features.shape[1]

        if n_others == 0:
            # No other agents — context is zeros
            context = torch.zeros(batch_size, self.embedding_dim,
                                  device=focal_features.device, dtype=focal_features.dtype)
        else:
            others_embeddings = self.agent_encoder(others_features)    # (batch, n_others, embedding_dim)

            # Attention: focal queries others
            query = self.attention_query(focal_embedding)              # (batch, embedding_dim)
            scale = math.sqrt(self.embedding_dim)
            scores = torch.bmm(
                others_embeddings, query.unsqueeze(2)
            ).squeeze(2) / scale                                       # (batch, n_others)
            weights = torch.softmax(scores, dim=1)                     # (batch, n_others)
            context = torch.bmm(
                weights.unsqueeze(1), others_embeddings
            ).squeeze(1)                                               # (batch, embedding_dim)

        # Decision network
        decision_input = torch.cat([fruit_level, focal_features, context], dim=1)
        return self.decision_net(decision_input)


def mutate_predictor_dims(
    embedding_dim: int,
    decision_hidden: int,
    prob: float = 0.3,
    min_dim: int = 4,
    max_dim: int = 64,
) -> tuple[int, int]:
    """Randomly shift AgentPredictor hyperparameters.

    Each dimension is independently mutated with probability `prob` by adding
    a random delta from {-2, -1, +1, +2}, clamped to [min_dim, max_dim].

    Args:
        embedding_dim: current embedding dimension
        decision_hidden: current decision network hidden size
        prob: probability of mutating each dimension (default 0.3)
        min_dim: minimum allowed dimension (default 4)
        max_dim: maximum allowed dimension (default 64)

    Returns:
        Tuple (new_embedding_dim, new_decision_hidden).
    """
    deltas = [-2, -1, +1, +2]

    new_emb = embedding_dim
    if np.random.random() < prob:
        new_emb = int(np.clip(embedding_dim + np.random.choice(deltas), min_dim, max_dim))

    new_dh = decision_hidden
    if np.random.random() < prob:
        new_dh = int(np.clip(decision_hidden + np.random.choice(deltas), min_dim, max_dim))

    return new_emb, new_dh


def transfer_predictor_weights(src: AgentPredictor, dst: AgentPredictor) -> None:
    """Copy overlapping weight slices from src to dst AgentPredictor.

    Handles size mismatches by copying only the overlapping [:min_dim] slice
    of each layer. Extra neurons in dst retain their random initialisation.
    Operates under torch.no_grad().

    Submodule transfer rules:
        agent_encoder[0]  Linear(2, emb):      copy [:min_emb, :2] weights
        attention_query   Linear(emb, emb):     copy [:min_emb, :min_emb] weights
        decision_net[0]   Linear(3+emb, dh):   copy [:min_dh, :3] (fixed) +
                                                     [:min_dh, 3:3+min_emb] (context)
        decision_net[2]   Linear(dh, 1):        copy [0, :min_dh] weights

    Args:
        src: source (parent) AgentPredictor
        dst: destination (child) AgentPredictor — modified in place
    """
    min_emb = min(src.embedding_dim, dst.embedding_dim)
    src_dh = src.decision_net[0].out_features
    dst_dh = dst.decision_net[0].out_features
    min_dh = min(src_dh, dst_dh)

    with torch.no_grad():
        # agent_encoder[0]: Linear(2, emb)
        dst.agent_encoder[0].weight.data[:min_emb, :2] = (
            src.agent_encoder[0].weight.data[:min_emb, :2]
        )
        dst.agent_encoder[0].bias.data[:min_emb] = (
            src.agent_encoder[0].bias.data[:min_emb]
        )

        # attention_query: Linear(emb, emb)
        dst.attention_query.weight.data[:min_emb, :min_emb] = (
            src.attention_query.weight.data[:min_emb, :min_emb]
        )
        dst.attention_query.bias.data[:min_emb] = (
            src.attention_query.bias.data[:min_emb]
        )

        # decision_net[0]: Linear(3 + emb, dh)
        dst.decision_net[0].weight.data[:min_dh, :3] = (
            src.decision_net[0].weight.data[:min_dh, :3]
        )
        dst.decision_net[0].weight.data[:min_dh, 3:3 + min_emb] = (
            src.decision_net[0].weight.data[:min_dh, 3:3 + min_emb]
        )
        dst.decision_net[0].bias.data[:min_dh] = (
            src.decision_net[0].bias.data[:min_dh]
        )

        # decision_net[2]: Linear(dh, 1)
        dst.decision_net[2].weight.data[0, :min_dh] = (
            src.decision_net[2].weight.data[0, :min_dh]
        )
        dst.decision_net[2].bias.data[0] = src.decision_net[2].bias.data[0]


@dataclass
class AgentGenome:
    """Stores an agent's neural architecture, model, and fitness for evolution.

    Attributes:
        agent_id: unique identifier for this genome
        embedding_dim: per-agent embedding size in AgentPredictor
        decision_hidden: hidden layer size in the decision network
        fitness: accumulated fitness score (e.g. food eaten this episode)
        nn_model: the AgentPredictor instance
        optimizer: Adam optimizer bound to nn_model
    """
    agent_id: int
    embedding_dim: int = 8
    decision_hidden: int = 16
    fitness: float = 0.0
    nn_model: AgentPredictor | None = None
    optimizer: torch.optim.Adam | None = None


def reproduce(
    parent_genomes: list[AgentGenome],
    food_eaten_counts: dict[int, int],
    foods_per_child: int,
) -> list[AgentGenome]:
    """Create the next generation of agents from parent genomes.

    Each parent produces floor(food_eaten / foods_per_child) children.
    Each child inherits mutated architecture dimensions and transferred weights
    from its parent, then gets a fresh Adam optimizer.

    Args:
        parent_genomes: list of AgentGenome for the current generation
        food_eaten_counts: dict mapping agent_id to food eaten this episode
        foods_per_child: food items required to produce one child

    Returns:
        List of child AgentGenomes with sequential IDs starting at 0.
        Empty list if no parent earned enough food.
    """
    children: list[AgentGenome] = []
    for parent in parent_genomes:
        n_children = food_eaten_counts.get(parent.agent_id, 0) // foods_per_child
        for _ in range(n_children):
            new_emb, new_dh = mutate_predictor_dims(parent.embedding_dim, parent.decision_hidden)
            child_model = AgentPredictor(embedding_dim=new_emb, decision_hidden=new_dh)
            if parent.nn_model is not None:
                transfer_predictor_weights(parent.nn_model, child_model)
            child_optimizer = torch.optim.Adam(child_model.parameters())
            children.append(AgentGenome(
                agent_id=len(children),
                embedding_dim=new_emb,
                decision_hidden=new_dh,
                nn_model=child_model,
                optimizer=child_optimizer,
            ))
    return children
