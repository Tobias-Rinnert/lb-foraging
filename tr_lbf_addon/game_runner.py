"""Game runner for LBF environment with neural network training.

This module provides GameRunner for orchestrating gameplay: initializing the environment,
resetting episodes, stepping through game loops, collecting metrics and rewards.
"""

import sys
import os

# Ensure tr_lbf_addon/ is on sys.path for bare imports in lbf_gym.py
_addon_dir = os.path.dirname(os.path.abspath(__file__))
if _addon_dir not in sys.path:
    sys.path.insert(0, _addon_dir)

import gymnasium as gym
from lbf_gym import LBF_GYM
from metrics_tracker import MetricsTracker


_env_counter = 0  # unique id per env registration to avoid re-registration errors


def default_params() -> dict:
    """Return default game parameters.

    Returns:
        Dict with keys: field_size, number_players, max_num_food, coop_mode,
        max_episode_steps, sight, min/max_player_level, min/max_food_level, penalty,
        fallback_to_closest.
    """
    return {
        "field_size": 20,
        "number_players": 5,
        "max_num_food": 8,
        "coop_mode": False,
        "max_episode_steps": 500,
        "sight": 10,
        "min_player_level": 1,
        "max_player_level": 5,
        "min_food_level": 1,
        "max_food_level": 5,
        "penalty": 0.0,
        "normalize_reward": True,
        "observe_agent_levels": True,
        "full_info_mode": True,
        "fallback_to_closest": False,
    }


class GameRunner:
    """Orchestrates LBF game episodes with neural network agent learning.

    Owns the gymnasium environment and LBF_GYM instance. Manages the game loop,
    collects rewards and metrics, coordinates agent learning.

    Call reset() after construction or rebuild() to start a new episode.
    Call step() each frame to advance by one timestep.
    """

    def __init__(self, params: dict) -> None:
        """Initialize the game runner.

        Args:
            params: game configuration dict (see default_params)
        """
        self.params: dict = dict(params)
        "Copy of game configuration parameters"
        self.env: gym.Env | None = None
        "Gymnasium environment instance"
        self.lbf_gym: LBF_GYM | None = None
        "LBF game state and logic manager"
        self.observation: tuple | None = None
        "Current gymnasium observation tuple [player_obs, ...]"
        self.step_count: int = 0
        "Number of steps taken in current episode"
        self.rewards: list[float] = []
        "Rewards collected in current episode (one per agent per step)"
        self.episode_over: bool = False
        "True when episode has reached terminal state"
        self.metrics: MetricsTracker = MetricsTracker()
        "Metrics aggregator (episode returns, NN losses, etc.)"
        self._cumulative_rewards: list[float] = []
        "Cumulative reward for each agent across episode"
        self._n_rows: int = params.get("number_players", 5)
        "Fixed-capacity number of rows in NN input vector"
        self._build_env()

    # -- public API ------------------------------------------------------------

    def reset(self) -> None:
        """Reset the environment to start a new episode.

        Initializes LBF_GYM from gymnasium observation, sets agent._n_rows, clears metrics.

        Returns:
            None (side effects on self.observation, self.lbf_gym, step_count, rewards, etc.)
        """
        self.observation, _ = self.env.reset(seed=None)
        self.lbf_gym = LBF_GYM(self.observation[0])
        for agent in self.lbf_gym.agents:
            agent._n_rows = self._n_rows
        self.step_count = 0
        self.rewards = [0.0] * self.params["number_players"]
        self._cumulative_rewards = [0.0] * self.params["number_players"]
        self.episode_over = False

    def step(self) -> None:
        """Advance the game by one timestep.

        Updates observations, coordinates agent actions, steps gymnasium, records rewards/metrics.
        No-op if episode is over.

        Returns:
            None (side effects on observation, rewards, step_count, episode_over, metrics)
        """
        if self.episode_over:
            return
        self.lbf_gym.update_observation(self.observation[0])
        actions = self.lbf_gym.agents_choose_actions(
            fallback_to_closest=self.params.get("fallback_to_closest", True),
        )
        self.observation, reward, terminated, truncated, _ = self.env.step(tuple(actions))
        self.rewards = list(reward)
        self.step_count += 1
        self.episode_over = bool(terminated or truncated)

        # Accumulate rewards for episode-level return metrics
        for i, r in enumerate(self.rewards):
            self._cumulative_rewards[i] += r

        # Record per-agent NN losses for this step
        self.metrics.record_step_losses(self.lbf_gym.last_step_losses_per_agent)

        # Finalise the episode metrics when the episode ends
        if self.episode_over:
            self.metrics.record_episode_end(self._cumulative_rewards)

    def rebuild(self, new_params: dict) -> None:
        """Close the current env, register a new one with new_params, and reset."""
        self.metrics.clear()
        if self.env is not None:
            self.env.close()
        self._n_rows = new_params.get("number_players", 5)
        self.params = dict(new_params)
        self._build_env()
        self.reset()

    # -- private ---------------------------------------------------------------

    def _build_env(self) -> None:
        """Register and initialize a gymnasium environment.

        Creates a unique environment ID, registers it with gymnasium, and stores
        the created environment in self.env.

        Returns:
            None (sets self.env side effect)
        """
        global _env_counter
        _env_counter += 1
        p = self.params
        coop_suffix = "-coop" if p["coop_mode"] else ""
        env_id = f"LBF-App-{_env_counter}x{p['field_size']}-{p['number_players']}p-{p['max_num_food']}f{coop_suffix}-v0"
        gym.register(
            id=env_id,
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": p["number_players"],
                "min_player_level": p["min_player_level"],
                "max_player_level": p["max_player_level"],
                "min_food_level": p["min_food_level"],
                "max_food_level": p["max_food_level"],
                "field_size": (p["field_size"], p["field_size"]),
                "max_num_food": p["max_num_food"],
                "sight": p["sight"],
                "max_episode_steps": p["max_episode_steps"],
                "force_coop": p["coop_mode"],
                "normalize_reward": p["normalize_reward"],
                "grid_observation": False,
                "observe_agent_levels": p["observe_agent_levels"],
                "penalty": p["penalty"],
                "render_mode": None,
                "full_info_mode": p["full_info_mode"],
            },
        )
        self.env = gym.make(env_id, disable_env_checker=True)
