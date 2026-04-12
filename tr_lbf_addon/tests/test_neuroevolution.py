"""Tests for tr_lbf_addon.neuroevolution — build_nn and add_agent_rows."""

import pytest
import torch
import torch.nn as nn
from tr_lbf_addon.neuroevolution import build_nn, add_agent_rows


# ── build_nn ─────────────────────────────────────────────────────────────────

class TestBuildNN:
    def test_output_shape(self):
        """Final output is (batch, 1) for any input."""
        model = build_nn(input_size=11, hidden_layers=[5])
        x = torch.randn(4, 11)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_in_range(self):
        """Sigmoid ensures output is in (0, 1)."""
        model = build_nn(input_size=11, hidden_layers=[5])
        x = torch.randn(100, 11)
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_multi_hidden(self):
        """Multiple hidden layers produce correct input/output shapes."""
        model = build_nn(input_size=11, hidden_layers=[8, 4])
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        # 3 linear layers: 11→8, 8→4, 4→1
        assert len(linears) == 3
        assert linears[0].in_features == 11
        assert linears[0].out_features == 8
        assert linears[1].in_features == 8
        assert linears[1].out_features == 4
        assert linears[2].out_features == 1


# ── add_agent_rows ────────────────────────────────────────────────────────────

class TestAddAgentRows:
    def _make_model(self, n_rows=3, hidden=5):
        """Helper: build a model for K=n_rows (input = 1 + n_rows*2)."""
        return build_nn(input_size=1 + n_rows * 2, hidden_layers=[hidden])

    def test_increases_input(self):
        """First Linear layer in_features increases by 2*n_new."""
        model = self._make_model(n_rows=3)
        old_in = [m for m in model.modules() if isinstance(m, nn.Linear)][0].in_features
        extended = add_agent_rows(model, n_new=2)
        new_in = [m for m in extended.modules() if isinstance(m, nn.Linear)][0].in_features
        assert new_in == old_in + 2 * 2

    def test_preserves_existing_weights(self):
        """Existing columns in the first Linear layer are unchanged."""
        model = self._make_model(n_rows=3)
        orig_first = [m for m in model.modules() if isinstance(m, nn.Linear)][0]
        old_weight = orig_first.weight.detach().clone()

        extended = add_agent_rows(model, n_new=1)
        new_first = [m for m in extended.modules() if isinstance(m, nn.Linear)][0]
        assert torch.allclose(new_first.weight[:, : old_weight.shape[1]], old_weight)

    def test_donor_cols_match_existing(self):
        """New columns are a copy of some donor row from the original weight matrix."""
        model = self._make_model(n_rows=3)
        orig_first = [m for m in model.modules() if isinstance(m, nn.Linear)][0]
        old_weight = orig_first.weight.detach().clone()
        K_old = (orig_first.in_features - 1) // 2

        extended = add_agent_rows(model, n_new=1)
        new_first = [m for m in extended.modules() if isinstance(m, nn.Linear)][0]
        new_cols = new_first.weight[:, old_weight.shape[1]:].detach()

        # New 2 columns must match one of the K_old existing agent-row pairs
        donor_match = any(
            torch.allclose(new_cols, old_weight[:, 1 + k * 2: 1 + k * 2 + 2])
            for k in range(K_old)
        )
        assert donor_match
