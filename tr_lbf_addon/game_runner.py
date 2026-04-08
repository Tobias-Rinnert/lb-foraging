import sys
import os

# Ensure tr_lbf_addon/ is on sys.path for bare imports in lbf_gym.py
_addon_dir = os.path.dirname(os.path.abspath(__file__))
if _addon_dir not in sys.path:
    sys.path.insert(0, _addon_dir)

import gymnasium as gym
from lbf_gym import LBF_GYM


_env_counter = 0  # unique id per env registration to avoid re-registration errors


def default_params() -> dict:
    """Return the default game parameters for a standard episode."""
    return {
        "field_size": 8,
        "number_players": 5,
        "max_num_food": 8,
        "coop_mode": False,
        "max_episode_steps": 50,
        "sight": 0,
        "min_player_level": 1,
        "max_player_level": 1,
        "min_food_level": 1,
        "max_food_level": 1,
        "penalty": 0.0,
        "normalize_reward": True,
        "observe_agent_levels": True,
        "full_info_mode": True,
        "fallback_to_closest": False,
    }


class GameRunner:
    """Model -- owns the gym env and LBF_GYM instance.

    Call reset() after construction or rebuild() to start a new episode.
    Call step() each frame to advance the game by one timestep.
    """

    def __init__(self, params: dict):
        self.params = dict(params)
        self.env = None
        self.lbf_gym = None
        self.observation = None
        self.step_count: int = 0
        self.rewards: list[float] = []
        self.episode_over: bool = False
        self._build_env()

    # -- public API ------------------------------------------------------------

    def reset(self):
        """Reset the current env to start a new episode."""
        self.observation, _ = self.env.reset(seed=None)
        self.lbf_gym = LBF_GYM(self.observation[0])
        self.step_count = 0
        self.rewards = [0.0] * self.params["number_players"]
        self.episode_over = False

    def step(self):
        """Advance the game by one timestep. No-op if episode is over."""
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

    def rebuild(self, new_params: dict):
        """Close the current env, register a new one with new_params, and reset."""
        if self.env is not None:
            self.env.close()
        self.params = dict(new_params)
        self._build_env()
        self.reset()

    # -- private ---------------------------------------------------------------

    def _build_env(self):
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
        self.env = gym.make(env_id)
