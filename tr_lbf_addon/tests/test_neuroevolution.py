"""Tests for tr_lbf_addon.neuroevolution — AgentPredictor attention-based model."""

import torch
from tr_lbf_addon.neuroevolution import AgentPredictor


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
