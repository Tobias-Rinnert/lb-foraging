from __future__ import annotations

import sys
import os

# Ensure tr_lbf_addon/ is on sys.path for bare imports in lbf_gym.py
_addon_dir = os.path.dirname(os.path.abspath(__file__))
if _addon_dir not in sys.path:
    sys.path.insert(0, _addon_dir)

import numpy as np
import gymnasium as gym
from lbf_gym import LBF_GYM
from metrics_tracker import MetricsTracker


_env_counter = 0  # unique id per env registration to avoid re-registration errors


def default_params() -> dict:
    """Return the default game parameters for a standard episode."""
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
        # survival / evolution params
        "hunger_rate": 0.001,
        "crowding_radius": 3,
        "crowding_penalty": 0.0002,
        "food_growth_rate": 0.005,
        "foods_per_child": 3,
        "grass_ratio": 0.70,
        "ca_smooth_iterations": 5,
    }


class GameRunner:
    """Model -- owns the gym env and LBF_GYM instance.

    Call reset() after construction or rebuild() to start a new episode.
    Call step() each frame to advance the game by one timestep.
    """

    def __init__(self, params: dict):
        self.params = dict(params)
        self.params["initial_number_players"] = params.get("number_players", 5)
        self.env = None
        self.lbf_gym = None
        self.observation = None
        self.step_count: int = 0
        self.rewards: list[float] = []
        self.episode_over: bool = False
        self.metrics = MetricsTracker()
        self._cumulative_rewards: list[float] = []
        # survival state
        self.ca_map: np.ndarray | None = None
        self.food_growth: dict[tuple, float] = {}
        self.agent_hunger: dict[int, float] = {}
        self.agent_food_eaten: dict[int, int] = {}
        self.dead_agents: set[int] = set()
        # neuroevolution state
        self._evolved_genomes: list = []  # list[AgentGenome]
        self._next_player_count: int = params.get("number_players", 5)
        self._build_env()

    # -- public API ------------------------------------------------------------

    def reset(self):
        """Reset to start a new episode. Injects evolved genomes into fresh agents."""
        # 1. Set player count from last evolution result
        self.params["number_players"] = self._next_player_count
        self._rebuild_env_if_player_count_changed()

        self.observation, _ = self.env.reset(seed=None)

        # 2. Generate CA map once per gameplay session (reset on rebuild)
        if self.ca_map is None:
            from map_generator import generate_ca_map
            self.ca_map = generate_ca_map(
                self.params["field_size"],
                grass_ratio=self.params.get("grass_ratio", 0.70),
                smooth_iterations=self.params.get("ca_smooth_iterations", 5),
            )

        self.lbf_gym = LBF_GYM(self.observation[0])

        # 3. Inject evolved genomes (NN weights + architecture) into new agents.
        #    If population size changed, input size changed too (input = n_agents*2+1).
        #    Rebuild with transfer_weights so no learned weights are lost.
        n_agents = len(self.lbf_gym.agents)
        new_input_size = n_agents * 2 + 1  # matches init_neural_network: (len(known_agents)+1)*2+1
        for i, genome in enumerate(self._evolved_genomes):
            if i < n_agents:
                a = self.lbf_gym.agents[i]
                a.nn_architecture = genome.hidden_layers
                import torch.nn as _nn
                old_linears = [m for m in genome.nn_model.modules() if isinstance(m, _nn.Linear)]
                old_input_size = old_linears[0].in_features if old_linears else new_input_size
                if old_input_size != new_input_size:
                    from neuroevolution import build_nn, transfer_weights
                    from torch.optim import Adam
                    new_model = build_nn(new_input_size, genome.hidden_layers)
                    transfer_weights(genome.nn_model, new_model)
                    a.neural_network = new_model
                    a.optimizer = Adam(new_model.parameters())
                else:
                    a.neural_network = genome.nn_model
                    a.optimizer = genome.optimizer

        # 4. Reset per-episode survival state
        self.step_count = 0
        self.rewards = [0.0] * self._next_player_count
        self._cumulative_rewards = [0.0] * self._next_player_count
        self.episode_over = False
        self.agent_hunger = {a.id: 0.0 for a in self.lbf_gym.agents}
        self.agent_food_eaten = {a.id: 0 for a in self.lbf_gym.agents}
        self.dead_agents = set()
        self.food_growth = {}  # populated on first step()

    def step(self):
        """Advance the game by one timestep. No-op if episode is over."""
        if self.episode_over:
            return

        # --- observation update (before env.step so agents can decide) --------
        self.lbf_gym.update_observation(
            self.observation[0],
            food_growth=self.food_growth,
            dead_agents=self.dead_agents,
            ca_map=self.ca_map,
        )

        actions = self.lbf_gym.agents_choose_actions(
            fallback_to_closest=self.params.get("fallback_to_closest", True),
            dead_agents=self.dead_agents,
        )

        self.observation, reward, terminated, truncated, _ = self.env.step(tuple(actions))
        self.rewards = list(reward)
        self.step_count += 1

        # --- food growth ------------------------------------------------------
        p = self.params
        # Read ALL food positions from the new raw observation (unfiltered, so we track
        # both ripe and unripe food for growth purposes).
        raw_field = self.observation[0]["field"]
        all_food_positions = {(int(r), int(c)) for r, c in zip(*np.where(raw_field > 0))}

        # remove tracking for eaten food (disappeared from field)
        for pos in list(self.food_growth.keys()):
            if pos not in all_food_positions:
                del self.food_growth[pos]

        # register new food positions (start at 0 so they must grow before agents see them)
        for pos in all_food_positions:
            if pos not in self.food_growth:
                self.food_growth[pos] = 0.0

        # grow food on grass cells, stone cells stay at 0
        if self.ca_map is not None:
            for pos in self.food_growth:
                r, c = pos
                if self.ca_map[r, c] == 1:
                    self.food_growth[pos] = min(1.0, self.food_growth[pos] + p.get("food_growth_rate", 0.005))
        else:
            # no CA map: all food immediately ripe
            for pos in self.food_growth:
                self.food_growth[pos] = 1.0

        # --- eating detection (reward > 0 → ate food) -------------------------
        alive_agents = [a for a in self.lbf_gym.agents if a.id not in self.dead_agents]
        for i, r in enumerate(self.rewards):
            if r > 0 and i not in self.dead_agents:
                self.agent_hunger[i] = 0.0
                self.agent_food_eaten[i] = self.agent_food_eaten.get(i, 0) + 1

        # --- hunger update ----------------------------------------------------
        for a in alive_agents:
            nearby = sum(
                1 for b in alive_agents if b.id != a.id
                and float(np.linalg.norm(np.array(a.position) - np.array(b.position)))
                <= p.get("crowding_radius", 3)
            )
            self.agent_hunger[a.id] = self.agent_hunger.get(a.id, 0.0) + (
                p.get("hunger_rate", 0.001) + nearby * p.get("crowding_penalty", 0.0002)
            )
            if self.agent_hunger[a.id] >= 1.0:
                self.dead_agents.add(a.id)
                a.is_alive = False

        # --- reward / metrics accumulation ------------------------------------
        for i, r in enumerate(self.rewards):
            if i < len(self._cumulative_rewards):
                self._cumulative_rewards[i] += r

        self.metrics.record_step_losses(self.lbf_gym.last_step_losses_per_agent)

        # --- episode termination ---------------------------------------------
        alive_count = len([a for a in self.lbf_gym.agents if a.id not in self.dead_agents])
        self.episode_over = bool(terminated or truncated or alive_count <= 1)

        if self.episode_over:
            self.metrics.record_episode_end(self._cumulative_rewards)

    def evolve(self) -> None:
        """Evolve agent NNs based on food eaten this episode. Call before reset()."""
        from neuroevolution import AgentGenome, reproduce

        if self.lbf_gym is None or not self.lbf_gym.agents:
            return

        alive = [a for a in self.lbf_gym.agents if a.id not in self.dead_agents]
        if not alive:
            # total extinction — restart with initial player count + fresh NNs
            self._evolved_genomes = []
            self._next_player_count = self.params.get("initial_number_players", 5)
            return

        first = alive[0]
        if first.neural_network is None or first.known_agents is None:
            # NNs never initialised (episode ended before first update_agents)
            self._next_player_count = max(1, len(alive))
            return

        input_size = (len(first.known_agents) + 1) * 2 + 1

        parent_genomes = [
            AgentGenome(
                agent_id=a.id,
                hidden_layers=list(a.nn_architecture),
                fitness=float(self.agent_food_eaten.get(a.id, 0)),
                nn_model=a.neural_network,
                optimizer=a.optimizer,
            )
            for a in alive
        ]

        children = reproduce(
            parent_genomes,
            self.agent_food_eaten,
            self.params.get("foods_per_child", 3),
            input_size,
        )

        if not children:
            # no one ate enough to reproduce — carry survivors forward with 1 child each
            from neuroevolution import mutate_architecture, build_nn, transfer_weights, Adam
            children = []
            for i, g in enumerate(parent_genomes):
                new_arch = mutate_architecture(g.hidden_layers)
                new_model = build_nn(input_size, new_arch)
                transfer_weights(g.nn_model, new_model)
                new_opt = Adam(new_model.parameters())
                children.append(
                    AgentGenome(
                        agent_id=i,
                        hidden_layers=new_arch,
                        fitness=0.0,
                        nn_model=new_model,
                        optimizer=new_opt,
                    )
                )

        self._evolved_genomes = children
        self._next_player_count = max(1, len(children))

    def rebuild(self, new_params: dict):
        """Close the current env, register a new one with new_params, and reset."""
        self.metrics.clear()
        self._evolved_genomes = []
        self.ca_map = None  # regenerate CA map with new params
        self._next_player_count = new_params.get("number_players", 5)
        if self.env is not None:
            self.env.close()
        self.params = dict(new_params)
        self.params["initial_number_players"] = new_params.get("number_players", 5)
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
        self.env = gym.make(env_id, disable_env_checker=True)

    def _rebuild_env_if_player_count_changed(self):
        """If the player count changed due to evolution, register a new env."""
        current_players = self.params.get("_last_registered_players")
        if current_players != self._next_player_count:
            self.params["_last_registered_players"] = self._next_player_count
            if self.env is not None:
                self.env.close()
            self._build_env()
