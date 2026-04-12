"""Neural network architecture for predicting agent fruit-targeting behavior.

Provides AgentPredictor, an attention-based architecture that processes variable
numbers of agents using shared weights and attention pooling to produce
permutation-invariant predictions while preserving n-th order beliefs.
"""

import copy
import math
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
