"""Tests for genome save/load round-trip in GameRunner.

Covers _save_best_genome, _load_saved_genome, and the first-reset genome injection
from a saved file. All tests use tmp_path and monkeypatch to redirect GENOME_SAVE_PATH
so no files are written to the repo root.
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

import game_runner as gr_module
from neuroevolution import AgentGenome, AgentPredictor


# ── Helpers ───────────────────────────────────────────────────────────

def _make_genome(agent_id: int = 0, fitness: float = 0.0,
                 embedding_dim: int = 8, decision_hidden: int = 16) -> AgentGenome:
    """Create an AgentGenome with a live AgentPredictor and Adam optimizer."""
    model = AgentPredictor(embedding_dim=embedding_dim, decision_hidden=decision_hidden)
    return AgentGenome(
        agent_id=agent_id,
        embedding_dim=embedding_dim,
        decision_hidden=decision_hidden,
        fitness=fitness,
        nn_model=model,
        optimizer=torch.optim.Adam(model.parameters()),
    )


def _make_runner(tmp_path, monkeypatch):
    """Create a GameRunner with mocked gymnasium and GENOME_SAVE_PATH redirected."""
    save_path = tmp_path / "saved_genome.pt"
    monkeypatch.setattr(gr_module, "GENOME_SAVE_PATH", save_path)
    params = gr_module.default_params()
    params["number_players"] = 2
    with patch("game_runner.gym") as mock_gym:
        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env
        runner = gr_module.GameRunner(params)
    return runner, save_path


# ── Tests ─────────────────────────────────────────────────────────────

class TestSaveGenome:
    """Tests for _save_best_genome."""

    def test_save_creates_file(self, tmp_path, monkeypatch):
        """_save_best_genome writes a file at GENOME_SAVE_PATH."""
        runner, save_path = _make_runner(tmp_path, monkeypatch)
        genome = _make_genome(agent_id=0, fitness=5.0)
        runner._save_best_genome([genome])
        assert save_path.exists()

    def test_save_picks_best_fitness(self, tmp_path, monkeypatch):
        """When given two genomes, the one with higher fitness is saved."""
        runner, save_path = _make_runner(tmp_path, monkeypatch)
        weak = _make_genome(agent_id=0, fitness=1.0, embedding_dim=4, decision_hidden=4)
        strong = _make_genome(agent_id=1, fitness=9.0, embedding_dim=12, decision_hidden=24)
        runner._save_best_genome([weak, strong])
        data = torch.load(save_path, weights_only=True)
        assert data["embedding_dim"] == 12
        assert data["decision_hidden"] == 24

    def test_save_noop_when_empty(self, tmp_path, monkeypatch):
        """_save_best_genome with an empty list does not create a file."""
        runner, save_path = _make_runner(tmp_path, monkeypatch)
        runner._save_best_genome([])
        assert not save_path.exists()

    def test_save_noop_when_no_model(self, tmp_path, monkeypatch):
        """_save_best_genome with a genome that has no nn_model does not create a file."""
        runner, save_path = _make_runner(tmp_path, monkeypatch)
        genome = AgentGenome(agent_id=0, fitness=5.0)  # nn_model=None
        runner._save_best_genome([genome])
        assert not save_path.exists()


class TestLoadGenome:
    """Tests for _load_saved_genome."""

    def test_load_returns_none_when_no_file(self, tmp_path, monkeypatch):
        """_load_saved_genome returns None when no save file exists."""
        runner, save_path = _make_runner(tmp_path, monkeypatch)
        assert not save_path.exists()
        result = runner._load_saved_genome()
        assert result is None

    def test_save_load_roundtrip_preserves_dims(self, tmp_path, monkeypatch):
        """Saved embedding_dim and decision_hidden survive a save/load round-trip."""
        runner, save_path = _make_runner(tmp_path, monkeypatch)
        genome = _make_genome(embedding_dim=12, decision_hidden=24, fitness=3.0)
        runner._save_best_genome([genome])
        loaded = runner._load_saved_genome()
        assert loaded is not None
        assert loaded.embedding_dim == 12
        assert loaded.decision_hidden == 24

    def test_save_load_roundtrip_preserves_weights(self, tmp_path, monkeypatch):
        """All layer weights of the nn_model survive a save/load round-trip."""
        runner, save_path = _make_runner(tmp_path, monkeypatch)
        genome = _make_genome(embedding_dim=8, decision_hidden=16, fitness=2.0)
        runner._save_best_genome([genome])
        loaded = runner._load_saved_genome()
        assert loaded.nn_model is not None
        for (name, orig_param), (_, loaded_param) in zip(
            genome.nn_model.named_parameters(),
            loaded.nn_model.named_parameters(),
        ):
            assert torch.allclose(orig_param, loaded_param), f"Mismatch in {name}"

    def test_loaded_genome_has_optimizer(self, tmp_path, monkeypatch):
        """Loaded genome has a fresh Adam optimizer attached."""
        runner, save_path = _make_runner(tmp_path, monkeypatch)
        genome = _make_genome(fitness=1.0)
        runner._save_best_genome([genome])
        loaded = runner._load_saved_genome()
        assert loaded.optimizer is not None


class TestFirstResetUsedSavedGenome:
    """Tests that a saved genome is used to seed the population on first reset."""

    def test_saved_genome_loaded_at_construction(self, tmp_path, monkeypatch):
        """When a save file exists, GameRunner loads it into _saved_genome at __init__."""
        save_path = tmp_path / "saved_genome.pt"
        monkeypatch.setattr(gr_module, "GENOME_SAVE_PATH", save_path)
        model = AgentPredictor(embedding_dim=8, decision_hidden=16)
        torch.save({
            "embedding_dim": 8,
            "decision_hidden": 16,
            "state_dict": model.state_dict(),
        }, save_path)
        params = gr_module.default_params()
        params["number_players"] = 2
        with patch("game_runner.gym") as mock_gym:
            mock_gym.make.return_value = MagicMock()
            runner = gr_module.GameRunner(params)
        assert runner._saved_genome is not None
        assert runner._saved_genome.embedding_dim == 8

    def test_first_reset_generates_evolved_genomes_from_save(self, tmp_path, monkeypatch):
        """First reset() with a save file populates _evolved_genomes and clears _saved_genome."""
        save_path = tmp_path / "saved_genome.pt"
        monkeypatch.setattr(gr_module, "GENOME_SAVE_PATH", save_path)
        model = AgentPredictor(embedding_dim=8, decision_hidden=16)
        torch.save({
            "embedding_dim": 8,
            "decision_hidden": 16,
            "state_dict": model.state_dict(),
        }, save_path)

        params = gr_module.default_params()
        params["number_players"] = 2
        params["foods_per_child"] = 3

        mock_agent_0 = MagicMock()
        mock_agent_0.id = 0
        mock_agent_1 = MagicMock()
        mock_agent_1.id = 1

        fake_obs = {
            "field": np.zeros((5, 5)),
            "player_infos": [
                {"id": 0, "position": (0, 0), "level": 1},
                {"id": 1, "position": (4, 4), "level": 2},
            ],
        }

        with patch("game_runner.gym") as mock_gym, \
             patch("game_runner.LBF_GYM") as mock_lbf_cls, \
             patch("game_runner.generate_ca_map") as mock_gen_map:

            mock_env = MagicMock()
            mock_gym.make.return_value = mock_env
            mock_env.reset.return_value = ([fake_obs], {})

            mock_gym_inst = MagicMock()
            mock_gym_inst.agents = [mock_agent_0, mock_agent_1]
            mock_gym_inst.fruits = []
            mock_lbf_cls.return_value = mock_gym_inst
            mock_gen_map.return_value = np.ones((5, 5), dtype=np.int8)

            runner = gr_module.GameRunner(params)
            assert runner._saved_genome is not None

            runner.reset()

        # saved genome spawns foods_per_child * n_players / foods_per_child = n_players children
        assert len(runner._evolved_genomes) > 0
        assert runner._saved_genome is None
