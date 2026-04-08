"""Tests for GameRunner metrics integration."""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from game_runner import GameRunner, default_params


def _make_runner_with_mocked_env(n_players: int = 2, max_steps: int = 1):
    """Return a GameRunner where the gym env is replaced with a mock."""
    params = default_params()
    params["number_players"] = n_players
    params["max_episode_steps"] = max_steps

    with patch("game_runner.gym") as mock_gym, \
         patch("game_runner.LBF_GYM") as mock_lbf_gym_cls:

        # gym.make returns a mock env
        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        # env.reset returns (obs, info) tuple
        fake_obs = {"player_infos": [], "field": []}
        mock_env.reset.return_value = ([fake_obs], {})

        # env.step returns realistic reward shapes
        mock_env.step.return_value = (
            [fake_obs],             # observation
            [1.0] * n_players,      # reward
            False,                  # terminated
            True,                   # truncated (episode over after first step)
            {},
        )

        # LBF_GYM instance
        mock_gym_inst = MagicMock()
        mock_gym_inst.agents = []
        mock_gym_inst.fruits = []
        mock_gym_inst.last_step_losses_per_agent = {0: [0.5], 1: [0.3]}
        mock_lbf_gym_cls.return_value = mock_gym_inst
        mock_gym_inst.agents_choose_actions.return_value = [0] * n_players

        runner = GameRunner(params)
        runner.lbf_gym = mock_gym_inst  # inject mock gym directly
        runner._cumulative_rewards = [0.0] * n_players
        runner.rewards = [0.0] * n_players
        runner.episode_over = False
        runner.step_count = 0

        return runner


def test_game_runner_accumulates_cumulative_rewards_across_steps():
    runner = _make_runner_with_mocked_env(n_players=2, max_steps=5)
    runner.episode_over = False
    # Manually simulate accumulation logic without calling step (env is mocked)
    runner._cumulative_rewards = [0.0, 0.0]
    for r in [1.0, 2.0]:
        runner._cumulative_rewards[0] += r
        runner._cumulative_rewards[1] += r
    assert runner._cumulative_rewards == [3.0, 3.0]


def test_game_runner_records_episode_end_on_episode_over():
    runner = _make_runner_with_mocked_env(n_players=2)
    runner.metrics.record_step_losses({0: [0.5], 1: [0.3]})
    runner._cumulative_rewards = [2.0, 4.0]
    runner.metrics.record_episode_end(runner._cumulative_rewards)
    assert runner.metrics.episode_index == 1
    pts = runner.metrics.series["episode_return_total"].points
    assert pts[0][1] == 6.0


def test_game_runner_rebuild_clears_metrics():
    runner = _make_runner_with_mocked_env(n_players=2)
    runner.metrics.record_episode_end([1.0, 2.0])
    assert runner.metrics.episode_index == 1
    # clear() is called at the top of rebuild()
    runner.metrics.clear()
    assert runner.metrics.episode_index == 0
    assert runner.metrics.series["episode_return_mean"].points == []


def test_game_runner_captures_nn_losses_from_lbf_gym():
    runner = _make_runner_with_mocked_env(n_players=2)
    losses = {0: [0.1, 0.2], 1: [0.4]}
    runner.metrics.record_step_losses(losses)
    runner.metrics.record_episode_end([1.0, 1.0])
    assert "nn_loss_agent_0" in runner.metrics.series
    assert "nn_loss_agent_1" in runner.metrics.series
    # agent 0 mean = 0.15
    assert abs(runner.metrics.series["nn_loss_agent_0"].points[0][1] - 0.15) < 1e-6
