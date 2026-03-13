# tr_lbf_addon/tests/test_app_game_runner.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from tr_lbf_addon.app.game_runner import GameRunner, default_params


def test_default_params_keys():
    params = default_params()
    required = {
        "field_size", "number_players", "max_num_food", "coop_mode",
        "max_episode_steps", "sight", "min_player_level", "max_player_level",
        "min_food_level", "max_food_level", "penalty",
        "normalize_reward", "observe_agent_levels", "full_info_mode",
    }
    assert required.issubset(set(params.keys()))


def test_runner_builds_and_resets():
    runner = GameRunner(default_params())
    runner.reset()
    assert runner.step_count == 0
    assert runner.episode_over is False
    assert runner.lbf_gym is not None


def test_runner_step_increments_counter():
    runner = GameRunner(default_params())
    runner.reset()
    runner.step()
    assert runner.step_count == 1


def test_runner_rewards_length_matches_players():
    params = default_params()
    runner = GameRunner(params)
    runner.reset()
    runner.step()
    assert len(runner.rewards) == params["number_players"]


def test_runner_rebuild_changes_field_size():
    runner = GameRunner(default_params())
    runner.reset()
    new_params = default_params()
    new_params["field_size"] = 10
    runner.rebuild(new_params)
    assert runner.params["field_size"] == 10
    assert runner.step_count == 0


def test_runner_episode_over_after_max_steps():
    params = default_params()
    params["max_episode_steps"] = 1
    runner = GameRunner(params)
    runner.reset()
    for _ in range(5):
        runner.step()
        if runner.episode_over:
            break
    assert runner.episode_over is True
