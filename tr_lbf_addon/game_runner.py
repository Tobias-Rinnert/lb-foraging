"""Game runner for LBF environment with neural network training and survival mechanics.

This module provides GameRunner for orchestrating gameplay: initializing the environment,
resetting episodes, stepping through game loops, collecting metrics, managing hunger/death,
food growth, and neuroevolution across episodes.
"""

import sys
import os
from pathlib import Path

# Ensure tr_lbf_addon/ is on sys.path for bare imports in lbf_gym.py
_addon_dir = os.path.dirname(os.path.abspath(__file__))
if _addon_dir not in sys.path:
    sys.path.insert(0, _addon_dir)

import numpy as np
import torch
import gymnasium as gym
from lbf_gym import LBF_GYM
from metrics_tracker import MetricsTracker
from map_generator import generate_ca_map
from neuroevolution import (
    AgentGenome,
    AgentPredictor,
    reproduce,
    mutate_predictor_dims,
    transfer_predictor_weights,
)

GENOME_SAVE_PATH = Path(__file__).parent.parent / "saved_genome.pt"
"Path where the best parent genome is persisted between runs."

_env_counter = 0  # unique id per env registration to avoid re-registration errors


def default_params() -> dict:
    """Return default game parameters.

    Returns:
        Dict with keys: field_size, number_players, max_num_food, coop_mode,
        max_episode_steps, sight, min/max_player_level, min/max_food_level, penalty,
        fallback_to_closest, and survival/evolution params.
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
        # Survival & evolution
        "min_level_1_food": 2,
        "hunger_rate": 0.001,
        "food_growth_rate": 0.005,
        "foods_per_child": 3,
        "grass_ratio": 0.70,
        "ca_smooth_iterations": 5,
    }


class GameRunner:
    """Orchestrates LBF game episodes with neural network agent learning and survival mechanics.

    Owns the gymnasium environment and LBF_GYM instance. Manages the game loop,
    collects rewards and metrics, coordinates agent learning, and drives neuroevolution
    between episodes (hunger, death, food growth, reproduction).

    Call reset() after construction or rebuild() to start a new episode.
    Call step() each frame to advance by one timestep.
    Call evolve() between episodes to produce the next generation.
    """

    def __init__(self, params: dict) -> None:
        """Initialize the game runner.

        Args:
            params: game configuration dict (see default_params)
        """
        self.params: dict = dict(params)
        "Copy of game configuration parameters"
        self.params["initial_number_players"] = params.get("number_players", 5)
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
        # Survival state
        self.ca_map: np.ndarray | None = None
        "Cellular automata terrain map (field_size x field_size): 0=stone, 1=grass"
        self.food_growth: dict[tuple, float] = {}
        "Maps fruit position tuple to growth value in [0, 1]; <1.0 means hidden"
        self.agent_hunger: dict[int, float] = {}
        "Maps agent id to hunger level in [0, 1]; >=1.0 means dead"
        self.agent_food_eaten: dict[int, int] = {}
        "Maps agent id to count of food items eaten this episode"
        self.dead_agents: set[int] = set()
        "Set of agent IDs that have died this episode"
        # Evolution state
        self._evolved_genomes: list[AgentGenome] = []
        "Child genomes from the last evolve() call, injected at next reset()"
        self._next_player_count: int = params.get("number_players", 5)
        "Number of players to register for the next episode"
        self._build_env()
        self._saved_genome: AgentGenome | None = self._load_saved_genome()
        "Best-parent genome loaded from disk at startup; None if no save file"

    # -- public API ------------------------------------------------------------

    def reset(self) -> None:
        """Reset the environment to start a new episode.

        1. Updates player count and rebuilds env if it changed.
        2. Generates CA terrain map on first call (reused across episodes).
        3. Initializes LBF_GYM and sets agent normalization bounds.
        4. Injects evolved genomes (or bootstraps from saved genome on first run).
        5. Resets survival state (hunger, food_eaten, dead_agents).
        6. Pre-populates food_growth with 1.0 for all initial fruits.

        Returns:
            None (side effects on self.observation, self.lbf_gym, step_count, etc.)
        """
        self.params["number_players"] = self._next_player_count
        self._rebuild_env_if_player_count_changed()

        # Generate terrain before env.reset() so spawn_food / spawn_players can
        # exclude stone cells via env.ca_map inside _is_empty_location.
        if self.ca_map is None:
            self.ca_map = generate_ca_map(
                self.params["field_size"],
                grass_ratio=self.params.get("grass_ratio", 0.70),
                smooth_iterations=self.params.get("ca_smooth_iterations", 5),
            )
        self.env.unwrapped.ca_map = self.ca_map

        self.observation, _ = self.env.reset(seed=None)

        self.lbf_gym = LBF_GYM(self.observation[0], ca_map=self.ca_map)
        for agent in self.lbf_gym.agents:
            agent._max_agent_level = self.params.get("max_player_level", 5)
            agent._max_fruit_level = self.params.get("max_food_level", 5)
            agent._max_distance = float(self.params.get("field_size", 20) * 2)

        # Genome injection priority:
        # 1. Evolved genomes from previous episode (normal inter-episode evolution)
        # 2. Saved genome from disk (first run after app start)
        # 3. Fresh start (no save file, first ever run)
        if self._evolved_genomes:
            self._inject_genomes(self._evolved_genomes)
        elif self._saved_genome is not None:
            foods_per_child = self.params.get("foods_per_child", 3)
            n = self.params.get("initial_number_players", self.params.get("number_players", 5))
            synthetic_food = {self._saved_genome.agent_id: foods_per_child * n}
            children = reproduce([self._saved_genome], synthetic_food, foods_per_child)
            self._evolved_genomes = children
            self._saved_genome = None
            self._inject_genomes(self._evolved_genomes)

        # Reset survival state
        self.agent_hunger = {a.id: 0.0 for a in self.lbf_gym.agents}
        self.agent_food_eaten = {a.id: 0 for a in self.lbf_gym.agents}
        self.dead_agents = set()

        # Pre-populate food_growth so all initial fruits are immediately visible
        self.food_growth = {tuple(f.position): 1.0 for f in self.lbf_gym.fruits}

        self.step_count = 0
        self.rewards = [0.0] * self.params["number_players"]
        self._cumulative_rewards = [0.0] * self.params["number_players"]
        self.episode_over = False

    def step(self) -> None:
        """Advance the game by one timestep.

        1. Updates observations (with food_growth filter, dead agents, ca_map).
        2. Coordinates agent actions (dead agents get 0).
        3. Steps gymnasium.
        4. Updates food_growth (new fruits start growing, eaten fruits removed).
        5. Resets hunger for agents that ate; increments food_eaten count.
        6. Increases hunger for alive agents; marks dead if hunger >= 1.0.
        7. Records rewards and NN losses.
        8. Terminates if env done or all agents dead.

        Returns:
            None (side effects on observation, rewards, step_count, episode_over, metrics)
        """
        if self.episode_over:
            return
        self.lbf_gym.update_observation(
            self.observation[0], self.food_growth, self.dead_agents, self.ca_map
        )
        actions = self.lbf_gym.agents_choose_actions(
            fallback_to_closest=self.params.get("fallback_to_closest", True),
            dead_agents=self.dead_agents,
        )
        self.observation, reward, terminated, truncated, _ = self.env.step(tuple(actions))
        self.rewards = list(reward)
        self.step_count += 1

        # Update food_growth from new observation field
        new_field = self.observation[0]["field"]
        rows, cols = np.where(new_field > 0)
        new_fruit_positions = {(int(r), int(c)) for r, c in zip(rows, cols)}
        old_positions = set(self.food_growth.keys())

        for pos in old_positions - new_fruit_positions:
            del self.food_growth[pos]  # fruit was eaten or disappeared
        for pos in new_fruit_positions - old_positions:
            self.food_growth[pos] = 0.0  # new fruit, starts growing

        food_growth_rate = self.params.get("food_growth_rate", 0.005)
        for pos in new_fruit_positions & old_positions:
            if self.ca_map is None or self.ca_map[pos] == 1:  # grass cell
                self.food_growth[pos] = min(1.0, self.food_growth[pos] + food_growth_rate)

        # Eating: reset hunger and count food for agents that received reward
        for agent, r in zip(self.lbf_gym.agents, self.rewards):
            if r > 0 and agent.id not in self.dead_agents:
                self.agent_hunger[agent.id] = 0.0
                self.agent_food_eaten[agent.id] = self.agent_food_eaten.get(agent.id, 0) + 1

        # Hunger: increase for alive agents; mark dead if >= 1.0
        hunger_rate = self.params.get("hunger_rate", 0.001)
        for agent in self.lbf_gym.agents:
            if agent.id in self.dead_agents:
                continue
            self.agent_hunger[agent.id] = (
                self.agent_hunger.get(agent.id, 0.0) + hunger_rate
            )
            if self.agent_hunger[agent.id] >= 1.0:
                self.dead_agents.add(agent.id)
                agent.is_alive = False

        # Accumulate rewards and record NN losses
        for i, r in enumerate(self.rewards):
            self._cumulative_rewards[i] += r
        self.metrics.record_step_losses(self.lbf_gym.last_step_losses_per_agent)

        # Terminate if env done or all agents dead
        alive_count = sum(1 for a in self.lbf_gym.agents if a.id not in self.dead_agents)
        self.episode_over = bool(terminated or truncated or alive_count == 0)
        if self.episode_over:
            self.metrics.record_episode_end(self._cumulative_rewards)

    def evolve(self) -> None:
        """Produce the next generation of agents from alive survivors.

        Builds parent genomes from alive agents, calls reproduce() to create children.
        Fallback: if no children but survivors exist, each survivor produces one mutated child.
        Total extinction: resets to initial player count and regenerates terrain.
        Saves the best parent genome to disk before setting evolved genomes.

        Returns:
            None (side effects on _evolved_genomes, _next_player_count, ca_map, _saved_genome)
        """
        parents: list[AgentGenome] = []
        for agent in (self.lbf_gym.agents if self.lbf_gym else []):
            if agent.id not in self.dead_agents and agent.neural_network is not None:
                parents.append(AgentGenome(
                    agent_id=agent.id,
                    embedding_dim=agent.embedding_dim,
                    decision_hidden=agent.decision_hidden,
                    fitness=float(self.agent_food_eaten.get(agent.id, 0)),
                    nn_model=agent.neural_network,
                    optimizer=agent.optimizer,
                ))

        foods_per_child = self.params.get("foods_per_child", 3)
        food_eaten_counts = {g.agent_id: int(g.fitness) for g in parents}
        children = reproduce(parents, food_eaten_counts, foods_per_child)

        # Fallback: survivors exist but none earned enough food for a child
        if not children and parents:
            for parent in parents:
                new_emb, new_dh = mutate_predictor_dims(
                    parent.embedding_dim, parent.decision_hidden
                )
                child_model = AgentPredictor(embedding_dim=new_emb, decision_hidden=new_dh)
                if parent.nn_model is not None:
                    transfer_predictor_weights(parent.nn_model, child_model)
                children.append(AgentGenome(
                    agent_id=len(children),
                    embedding_dim=new_emb,
                    decision_hidden=new_dh,
                    nn_model=child_model,
                    optimizer=torch.optim.Adam(child_model.parameters()),
                ))

        if not children:
            # Total extinction: start fresh
            self._evolved_genomes = []
            self._next_player_count = self.params.get(
                "initial_number_players", self.params.get("number_players", 5)
            )
            self.ca_map = None  # regenerate terrain next episode
        else:
            self._save_best_genome(parents)
            self._evolved_genomes = children
            self._next_player_count = len(children)

    def rebuild(self, new_params: dict) -> None:
        """Close the current env, register a new one with new_params, and reset.

        Clears evolved genomes, resets terrain, and updates initial player count.
        Does NOT clear _saved_genome — it persists across rebuilds.
        """
        self.metrics.clear()
        if self.env is not None:
            self.env.close()
        self.params = dict(new_params)
        self.params["initial_number_players"] = new_params.get("number_players", 5)
        self._evolved_genomes = []
        self.ca_map = None
        self._next_player_count = new_params.get("number_players", 5)
        # _saved_genome intentionally NOT cleared — survives settings changes
        self._build_env()
        self.reset()

    # -- private ---------------------------------------------------------------

    def _build_env(self) -> None:
        """Register and initialize a gymnasium environment.

        Creates a unique environment ID, registers it with gymnasium, and stores
        the created environment in self.env. Records the registered player count
        in _last_registered_players for change detection.

        Returns:
            None (sets self.env and self._last_registered_players as side effects)
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
                "min_level_1_food": p["min_level_1_food"],
            },
        )
        self.env = gym.make(env_id, disable_env_checker=True)
        self._last_registered_players: int = p["number_players"]

    def _rebuild_env_if_player_count_changed(self) -> None:
        """Close and rebuild the gymnasium env if the player count has changed.

        Compares _last_registered_players against _next_player_count. If different,
        closes the current env and calls _build_env() with the updated params.

        Returns:
            None (side effects on self.env and self._last_registered_players)
        """
        if getattr(self, "_last_registered_players", None) != self._next_player_count:
            if self.env is not None:
                self.env.close()
            self._build_env()

    def _inject_genomes(self, genomes: list[AgentGenome]) -> None:
        """Inject evolved genome architecture and weights into agents by position.

        Zips agents with genomes in order, setting embedding_dim, decision_hidden,
        neural_network, and optimizer on each agent. Agents without a matching genome
        (if len(genomes) < len(agents)) retain their defaults.

        Args:
            genomes: list of AgentGenome to inject into agents
        """
        for agent, genome in zip(self.lbf_gym.agents, genomes):
            agent.embedding_dim = genome.embedding_dim
            agent.decision_hidden = genome.decision_hidden
            agent.neural_network = genome.nn_model
            agent.optimizer = genome.optimizer

    def _save_best_genome(self, parent_genomes: list[AgentGenome]) -> None:
        """Persist the highest-fitness parent genome to disk.

        Saves embedding_dim, decision_hidden, and nn_model state_dict to
        GENOME_SAVE_PATH using torch.save. Silent no-op if parent_genomes is
        empty or the best parent has no model.

        Args:
            parent_genomes: list of parent AgentGenomes from this episode
        """
        if not parent_genomes:
            return
        best = max(parent_genomes, key=lambda g: g.fitness)
        if best.nn_model is None:
            return
        torch.save(
            {
                "embedding_dim": best.embedding_dim,
                "decision_hidden": best.decision_hidden,
                "state_dict": best.nn_model.state_dict(),
            },
            GENOME_SAVE_PATH,
        )

    def _load_saved_genome(self) -> AgentGenome | None:
        """Load a genome from disk if GENOME_SAVE_PATH exists.

        Reconstructs an AgentPredictor with the saved dims, loads its state dict,
        and wraps it in a fresh AgentGenome with a new Adam optimizer.

        Returns:
            AgentGenome if the save file exists, None otherwise.
        """
        if not GENOME_SAVE_PATH.exists():
            return None
        data = torch.load(GENOME_SAVE_PATH, weights_only=True)
        model = AgentPredictor(
            embedding_dim=data["embedding_dim"],
            decision_hidden=data["decision_hidden"],
        )
        model.load_state_dict(data["state_dict"])
        return AgentGenome(
            agent_id=0,
            embedding_dim=data["embedding_dim"],
            decision_hidden=data["decision_hidden"],
            nn_model=model,
            optimizer=torch.optim.Adam(model.parameters()),
        )
