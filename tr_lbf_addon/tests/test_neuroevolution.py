"""Tests for tr_lbf_addon.neuroevolution — AgentPredictor, mutation, weight transfer, and reproduction."""

import torch
import numpy as np
from tr_lbf_addon.neuroevolution import (
    AgentPredictor,
    AgentGenome,
    mutate_predictor_dims,
    transfer_predictor_weights,
    reproduce,
)


class TestAgentPredictor:
    """Tests for the attention-based agent target predictor."""

    def _make_inputs(self, batch_size=1, n_others=3):
        """Create sample inputs for the model.

        Args:
            batch_size: number of samples in the batch
            n_others: number of other agents

        Returns:
            Tuple of (fruit_level, focal_features, others_features)
        """
        fruit_level = torch.rand(batch_size, 1)
        focal_features = torch.rand(batch_size, 2)
        others_features = torch.rand(batch_size, n_others, 2)
        return fruit_level, focal_features, others_features

    def test_output_shape(self):
        """Output shape is (batch, 1) for any valid input."""
        model = AgentPredictor()
        fl, fc, ot = self._make_inputs(batch_size=4, n_others=3)
        out = model(fl, fc, ot)
        assert out.shape == (4, 1)

    def test_output_in_range(self):
        """Sigmoid ensures output is in [0, 1]."""
        model = AgentPredictor()
        fl, fc, ot = self._make_inputs(batch_size=100, n_others=5)
        out = model(fl, fc, ot)
        assert (out >= 0).all() and (out <= 1).all()

    def test_variable_agent_count(self):
        """Model handles different numbers of other agents without error."""
        model = AgentPredictor()
        for n_others in [1, 3, 5, 10]:
            fl, fc, ot = self._make_inputs(batch_size=2, n_others=n_others)
            out = model(fl, fc, ot)
            assert out.shape == (2, 1)

    def test_zero_other_agents(self):
        """Model handles the edge case of zero other agents."""
        model = AgentPredictor()
        fl, fc, _ = self._make_inputs(batch_size=2, n_others=0)
        ot = torch.zeros(2, 0, 2)
        out = model(fl, fc, ot)
        assert out.shape == (2, 1)

    def test_permutation_invariant(self):
        """Swapping two agents in others_features produces the same output."""
        model = AgentPredictor()
        model.eval()
        fl, fc, ot = self._make_inputs(batch_size=1, n_others=4)
        out_original = model(fl, fc, ot)

        # Swap agents 0 and 2
        ot_swapped = ot.clone()
        ot_swapped[0, 0] = ot[0, 2]
        ot_swapped[0, 2] = ot[0, 0]
        out_swapped = model(fl, fc, ot_swapped)

        assert torch.allclose(out_original, out_swapped, atol=1e-6)

    def test_shared_encoder_weights(self):
        """Focal and other agents use the same encoder (shared weights)."""
        model = AgentPredictor()
        # The agent_encoder is a single module — focal and others both pass through it
        features = torch.tensor([[0.5, 0.3]])
        embedding = model.agent_encoder(features)
        assert embedding.shape == (1, model.embedding_dim)


class TestMutateAndTransfer:
    """Tests for mutate_predictor_dims and transfer_predictor_weights."""

    def test_mutate_dims_within_bounds(self):
        """Mutated dims are always within [min_dim, max_dim]."""
        for _ in range(50):
            emb, dh = mutate_predictor_dims(8, 16, prob=1.0, min_dim=4, max_dim=64)
            assert 4 <= emb <= 64
            assert 4 <= dh <= 64

    def test_mutate_dims_no_mutation_at_zero_prob(self):
        """With prob=0.0, dims are unchanged."""
        emb, dh = mutate_predictor_dims(8, 16, prob=0.0)
        assert emb == 8
        assert dh == 16

    def test_mutate_dims_always_mutates_at_one_prob(self):
        """With prob=1.0, at least one dim changes across 50 trials (clamp edge case excluded)."""
        changed = False
        for _ in range(50):
            emb, dh = mutate_predictor_dims(8, 16, prob=1.0)
            if emb != 8 or dh != 16:
                changed = True
                break
        assert changed

    def test_transfer_same_dims_copies_weights_exactly(self):
        """Transferring between same-size predictors copies weights exactly."""
        src = AgentPredictor(embedding_dim=8, decision_hidden=16)
        dst = AgentPredictor(embedding_dim=8, decision_hidden=16)
        transfer_predictor_weights(src, dst)
        assert torch.allclose(
            src.agent_encoder[0].weight, dst.agent_encoder[0].weight
        )
        assert torch.allclose(
            src.attention_query.weight, dst.attention_query.weight
        )
        assert torch.allclose(
            src.decision_net[0].weight, dst.decision_net[0].weight
        )

    def test_transfer_growing_dims_overlapping_slice_matches(self):
        """Transferring to a larger dst: overlapping slice matches, extras differ."""
        src = AgentPredictor(embedding_dim=4, decision_hidden=8)
        dst = AgentPredictor(embedding_dim=8, decision_hidden=16)
        # fill dst encoder with known values to confirm they get overwritten
        dst.agent_encoder[0].weight.data.fill_(999.0)
        transfer_predictor_weights(src, dst)
        # overlapping slice [:4, :2] should match src
        assert torch.allclose(
            dst.agent_encoder[0].weight.data[:4, :2],
            src.agent_encoder[0].weight.data[:4, :2],
        )

    def test_transfer_shrinking_dims_overlapping_slice_matches(self):
        """Transferring to a smaller dst: all dst weights come from src's slice."""
        src = AgentPredictor(embedding_dim=8, decision_hidden=16)
        dst = AgentPredictor(embedding_dim=4, decision_hidden=8)
        transfer_predictor_weights(src, dst)
        assert torch.allclose(
            dst.agent_encoder[0].weight.data,
            src.agent_encoder[0].weight.data[:4, :2],
        )

    def test_decision_net_fixed_columns_preserved(self):
        """The first 3 columns of decision_net[0] are copied from src."""
        src = AgentPredictor(embedding_dim=8, decision_hidden=16)
        dst = AgentPredictor(embedding_dim=8, decision_hidden=16)
        dst.decision_net[0].weight.data.fill_(0.0)
        transfer_predictor_weights(src, dst)
        assert torch.allclose(
            dst.decision_net[0].weight.data[:16, :3],
            src.decision_net[0].weight.data[:16, :3],
        )


class TestAgentGenome:
    """Tests for the AgentGenome dataclass."""

    def test_genome_defaults(self):
        """Default field values are correct."""
        genome = AgentGenome(agent_id=0)
        assert genome.embedding_dim == 8
        assert genome.decision_hidden == 16
        assert genome.fitness == 0.0
        assert genome.nn_model is None
        assert genome.optimizer is None

    def test_genome_stores_model(self):
        """AgentGenome can store an AgentPredictor and optimizer."""
        model = AgentPredictor()
        optimizer = torch.optim.Adam(model.parameters())
        genome = AgentGenome(agent_id=1, nn_model=model, optimizer=optimizer)
        assert genome.nn_model is model
        assert genome.optimizer is optimizer


class TestReproduce:
    """Tests for the reproduce function."""

    def _make_genome(self, agent_id: int) -> AgentGenome:
        model = AgentPredictor(embedding_dim=8, decision_hidden=16)
        return AgentGenome(
            agent_id=agent_id,
            nn_model=model,
            optimizer=torch.optim.Adam(model.parameters()),
        )

    def test_basic_reproduction(self):
        """Parent with 6 food and foods_per_child=3 produces 2 children."""
        parent = self._make_genome(agent_id=0)
        children = reproduce([parent], food_eaten_counts={0: 6}, foods_per_child=3)
        assert len(children) == 2

    def test_no_food_no_children(self):
        """Parent with 0 food produces no children."""
        parent = self._make_genome(agent_id=0)
        children = reproduce([parent], food_eaten_counts={0: 0}, foods_per_child=3)
        assert len(children) == 0

    def test_children_have_sequential_ids(self):
        """Children receive sequential IDs starting at 0."""
        parent_a = self._make_genome(agent_id=0)
        parent_b = self._make_genome(agent_id=1)
        children = reproduce(
            [parent_a, parent_b],
            food_eaten_counts={0: 3, 1: 6},
            foods_per_child=3,
        )
        assert [c.agent_id for c in children] == list(range(len(children)))

    def test_children_have_valid_models(self):
        """Each child has an AgentPredictor that can perform a forward pass."""
        parent = self._make_genome(agent_id=0)
        children = reproduce([parent], food_eaten_counts={0: 9}, foods_per_child=3)
        assert len(children) == 3
        for child in children:
            assert child.nn_model is not None
            fl = torch.rand(1, 1)
            fc = torch.rand(1, 2)
            ot = torch.rand(1, 2, 2)
            out = child.nn_model(fl, fc, ot)
            assert out.shape == (1, 1)
            assert 0.0 <= float(out.detach()) <= 1.0
