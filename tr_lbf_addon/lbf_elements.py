"""LBF agent and fruit representations, and agent cognition logic.

This module implements the core game entities and agent decision-making:
- Fruit: representation of collectible fruit on the map
- Agent: player entity with pathfinding, neural network prediction, and learning
- _build_nn_input: builds structured NN inputs for the attention-based AgentPredictor

The Agent class handles fruit selection, cooperative target prediction, path planning,
and neural network training. It integrates with the pathfinding, NN prediction,
and expected-reward selection systems.
"""

import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import torch
import torch.nn as nn



def _build_nn_input(focal_pos, focal_level, others, fruit, grid,
                    max_agent_level, max_fruit_level, max_distance):
    """Build structured NN input with absolute normalization.

    Returns three separate arrays for the attention-based AgentPredictor:
    fruit_level, focal_features, and others_features. All values normalized
    against fixed game-settings bounds.

    Args:
        focal_pos: np.ndarray position of the agent being predicted
        focal_level: int level of the agent being predicted
        others: list of dicts with keys "id", "position", and "level" (all other agents)
        fruit: Fruit object (target to compute distance to)
        grid: np.ndarray pathfinding grid (1=walkable, 0=obstacle)
        max_agent_level: int maximum possible agent level from game settings
        max_fruit_level: int maximum possible fruit level from game settings
        max_distance: float maximum possible A* path length (e.g. field_size * 2)

    Returns:
        tuple of three np.ndarrays:
            fruit_level: shape (1,) — normalized fruit level
            focal_features: shape (2,) — [normalized level, normalized distance]
            others_features: shape (n_others, 2) — each row [normalized level, normalized distance]
    """
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

    def _dist(pos, free_slots):
        if not free_slots:
            return max_distance
        slot = min(free_slots, key=lambda s: np.linalg.norm(s - np.array(pos)))
        g = Grid(matrix=grid.tolist())
        path, _ = finder.find_path(
            g.node(int(pos[1]), int(pos[0])),
            g.node(int(slot[1]), int(slot[0])),
            g,
        )
        return float(len(path)) if path else max_distance

    fruit_level = np.array(
        [fruit.level / max_fruit_level if max_fruit_level > 0 else 0.0],
        dtype=np.float32,
    )

    focal_dist = _dist(focal_pos, fruit.free_slots)
    focal_features = np.array([
        focal_level / max_agent_level if max_agent_level > 0 else 0.0,
        focal_dist / max_distance if max_distance > 0 else 0.0,
    ], dtype=np.float32)

    n_others = len(others)
    if n_others == 0:
        others_features = np.zeros((0, 2), dtype=np.float32)
    else:
        others_features = np.empty((n_others, 2), dtype=np.float32)
        for i, agent in enumerate(others):
            dist = _dist(agent["position"], fruit.free_slots)
            others_features[i, 0] = agent["level"] / max_agent_level if max_agent_level > 0 else 0.0
            others_features[i, 1] = dist / max_distance if max_distance > 0 else 0.0

    return fruit_level, focal_features, others_features


@dataclass
class Fruit:
    """Represents a collectible fruit on the game map.

    Attributes:
        position: (row, col) location of the fruit on the grid
        level: the fruit's level; determines solo or cooperative loadability
        free_slots: list of adjacent (row, col) positions where agents can stand to load this fruit
    """
    position: np.ndarray
    "The (row, col) position of the fruit on the game map"
    level: int
    "The fruit's level; higher levels require cooperative loading"
    free_slots: list[np.ndarray] | None = None
    "Adjacent positions where agents can stand to load this fruit"



class Agent:
    """Represents a player agent with cognition, pathfinding, and neural network learning.

    An Agent maintains its state (position, level), perceives the game world (known agents/fruits),
    plans paths via A*, predicts other agents' targets using a neural network, and learns from
    ground-truth labels. It uses combinatorial expected-reward logic to select targets.
    """

    def __init__(self, id: int,
                 position: np.ndarray,
                 level: int) -> None:
        """Initialize an agent.

        Args:
            id: unique agent identifier
            position: starting (row, col) position on the map
            level: starting cooperation level (contribution to cooperative loads)
        """

        self.position: np.ndarray = position
        "Current (row, col) position on the map"
        self.level: int = level
        "Cooperation level (contribution to cooperative loads)"
        self.id: int = id
        "Unique agent identifier"

        self.round_counter: int = 0
        "Current game step/round number"
        self.known_fruits: list[Fruit] | None = None
        "Fruits the agent is aware of (from observation)"
        self.known_agents: list[dict] | None = None
        "Information about other agents: {id, position, level, position_history, last_action, is_loading}"
        self.target: Fruit | None = None
        "Currently selected target fruit to move toward"
        self.position_history: list[np.ndarray] = []
        "History of past positions for movement analysis"
        self.last_action: np.int64 | None = None
        "Last action executed (0=none, 1-4=directions, 5=load)"
        self.memory_size: int = 3
        "Number of past steps to remember for direction inference"
        self.is_loading: bool = False
        "True if agent is currently at a fruit's loading slot loading it"
        self.path_goal: np.ndarray | None = None
        "Target (row, col) position to reach on the map (usually fruit slot)"
        self.path_finding_grid: np.ndarray | None = None
        "Occupancy grid for A* pathfinding (1=walkable, 0=obstacle)"
        self.current_path: np.ndarray | None = None
        "Current planned path as array of (row, col) positions"
        self.pathfinding_alg: AStarFinder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        "A* pathfinding algorithm instance"
        self.is_alive: bool = True
        "True while the agent is alive; dead agents are skipped for cognition and learning"
        self.embedding_dim: int = 8
        "AgentPredictor embedding dimension; may be set from an evolved genome before init_neural_network"
        self.decision_hidden: int = 16
        "AgentPredictor decision hidden size; may be set from an evolved genome before init_neural_network"
        self.optimizer: torch.optim.Optimizer | None = None
        "Adam optimizer for NN training"
        self._max_agent_level: int = 5
        "Maximum possible agent level from game settings, used for absolute normalization"
        self._max_fruit_level: int = 5
        "Maximum possible fruit level from game settings, used for absolute normalization"
        self._max_distance: float = 40.0
        "Maximum possible A* path length (field_size * 2), used for absolute normalization"
        self.neural_network: nn.Sequential | None = None
        "Neural network model for predicting other agents' targets"
        self.predictions: list[dict] = []
        "Predictions made by the NN: {round, agent_id, trainings_data, prediction, ground_truth, fruit_pos}"
        self.predicted_targets: dict[int, Fruit] = {}
        "Mapping from agent_id to their predicted target fruit"
        self.predicted_paths: dict[int, np.ndarray] = {}
        "Mapping from agent_id to their predicted A* path"
        self.prediction_round: dict[int, int] = {}
        "Mapping from agent_id to the round when the prediction was made"


    def __repr__(self) -> str:
        """Return a human-readable representation of the agent state."""
        return f"id: {self.id}, position: {self.position}, level: {self.level}, current target: {self.target}, is loading: {self.is_loading}"
    

    """Information processing"""
    
    def process_agent_infos(self, agents: list["Agent"]) -> None:
        """Perceive and store information about other agents.

        For each agent other than self, extracts id, position, level, position history,
        last action, and loading status. Position history is trimmed to memory_size.

        Args:
            agents: list of all Agent objects (including self)

        Returns:
            None (sets self.known_agents side effect)
        """
        self.known_agents = [{"id": agent.id,
                                   "position": agent.position,
                                   "level": agent.level,
                                   "position_history": agent.position_history[-self.memory_size:],
                                   "last_action": agent.last_action,
                                   "is_loading": agent.is_loading
                                   } for agent in agents if agent.id != self.id]
    
    
    """Agent chooses action logic"""
    
    def choose_next_action(self) -> np.int64:
        """
        Choose between walking directions or loading a fruit.

        Returns:
            np.int64: the next action (0=none, 1-4=directions, 5=load)
        """
        if not self.is_alive:
            return np.int64(0)

        if not self.target.free_slots:
            self.target = None  # force re-selection next step
            return np.int64(0)

        # Exclude slots already occupied by other agents so agents spread out
        other_positions = {tuple(a["position"]) for a in (self.known_agents or [])}
        available_slots = [s for s in self.target.free_slots if tuple(s) not in other_positions]
        if not available_slots:
            self.target = None  # all slots taken, try a different fruit next step
            return np.int64(0)

        # pick the closest unoccupied slot
        self.path_goal = min(available_slots, key=lambda x: np.linalg.norm(x - self.position))

        # if the current position is the current_path_goal, the player is going to load
        if np.all(self.position == self.path_goal):
            return self.action_string_to_int("load")

        self.current_path = self.get_path(self.position, self.path_goal)

        # no path found, force re-selection next step rather than idling permanently
        if self.current_path is None:
            self.target = None
            return np.int64(0)

        # get the next position in the path given the current position
        next_position = self.current_path[np.where(np.all(self.current_path == self.position, axis=1))[0][0] + 1]
        pos_diff = next_position - self.position
        direction = self.positional_difference_to_direction(pos_diff)
        return self.action_string_to_int(direction)


    def positional_difference_to_direction(self, positional_difference: np.ndarray) -> str:
        """
        Translate a given positional_difference to a direction

        Args:
            positional_difference (np.array): the positional difference to the next position

        Returns:
            str: the direction as a string 
        """               
        if np.all(positional_difference == np.array([-1, 0])):
            return "north"
        elif np.all(positional_difference == np.array([1, 0])):
            return "south"
        elif np.all(positional_difference == np.array([0, -1])):
            return "west"
        elif np.all(positional_difference == np.array([0, 1])):
            return "east"
        else:
            return "no move"

    def action_string_to_int(self, action: str) -> np.int64:
        """
        Convert the action string to the according np.int64 from the environment

        Args:
            action (str): The action as a string (north, south, west, east, load)

        Returns:
            (np.int64): The action as np.int64
        """
        match action:
            case "north":
                return np.int64(1)
            case "south":
                return np.int64(2)
            case "west":
                return np.int64(3)
            case "east":
                return np.int64(4)
            case "load":
                return np.int64(5)
            case _:
                return np.int64(0)

    """Target selection"""
    
    def choose_fruit(self) -> None:
        """Select a target fruit using conditional re-prediction and expected reward.

        Predictions persist across timesteps. Re-prediction only occurs when an agent deviates
        from its predicted path, which drastically reduces NN calls. Own target selection uses
        combinatorial expected reward via select_fruit_by_expected_reward().

        Steps:
            1. Skip re-selection if current target is still on the map.
            2. Build feasible fruit list (solo + cooperative).
            3. Invalidate stale predictions for fruits no longer on the map.
            4. For each other agent: skip NN call if still on predicted path, else re-predict.
            5. Select own target by expected reward.

        Returns:
            None (sets self.target side effect)
        """
        fruit_positions = np.array([fruit.position for fruit in self.known_fruits])
        current_target_still_in_game = (
            self.target is not None
            and np.any(np.all(self.target.position == fruit_positions, axis=1))
        )

        if current_target_still_in_game:
            return

        coop_levels = self.get_possible_coop_level_sums([agent["level"] for agent in self.known_agents])
        feasible_fruits = [
            fruit for fruit in self.known_fruits
            if fruit.level in coop_levels or fruit.level <= self.level
        ]

        # Invalidate predictions for fruits that are no longer on the map
        for agent_id in list(self.predicted_targets.keys()):
            predicted_fruit = self.predicted_targets[agent_id]
            if not np.any(np.all(predicted_fruit.position == fruit_positions, axis=1)):
                del self.predicted_targets[agent_id]
                del self.predicted_paths[agent_id]
                del self.prediction_round[agent_id]

        # For each other agent: skip NN call if still on predicted path, else re-predict
        for agent_info in self.known_agents:
            agent_id = agent_info["id"]
            agent_position = agent_info["position"]
            if self.is_agent_on_predicted_path(agent_id, agent_position):
                continue
            predicted_target = self.predict_agent_target(agent_id, feasible_fruits)
            if predicted_target is not None:
                self.compute_predicted_path(agent_id, agent_position, predicted_target)

        self.target = self.select_fruit_by_expected_reward(feasible_fruits)


    def is_agent_on_predicted_path(self, agent_id: int, position: np.ndarray) -> bool:
        """Check if the agent is still on its predicted path and trim the path to the current position.

        Returns True if the position is found anywhere on the remaining predicted path (handles
        collision delays by not requiring timestep-exact matching). Trims the stored path to
        start at the found position so subsequent calls advance the cursor.

        Args:
            agent_id (int): ID of the agent to check
            position (np.ndarray): current position of the agent

        Returns:
            bool: True if the agent is on the predicted path, False otherwise
        """
        if agent_id not in self.predicted_paths:
            return False
        path = self.predicted_paths[agent_id]
        for i, step in enumerate(path):
            if np.array_equal(step, position):
                self.predicted_paths[agent_id] = path[i:]
                return True
        return False


    def predict_agent_target(self, agent_id: int, feasible_fruits: list) -> "Fruit":
        """Predict which fruit the given agent will target and return the most probable fruit.

        Uses _build_row_input to assemble a fixed-capacity row-based NN input.
        Stores all predictions in self.predictions.

        Args:
            agent_id (int): ID of the agent to predict for
            feasible_fruits (list[Fruit]): fruits the agent could feasibly target

        Returns:
            Fruit: the fruit with the highest predicted probability
        """
        if self.neural_network is None:
            return feasible_fruits[0] if feasible_fruits else None

        focal_info = next((a for a in (self.known_agents or []) if a["id"] == agent_id), None)
        if focal_info is None:
            return feasible_fruits[0] if feasible_fruits else None

        others = [a for a in (self.known_agents or []) if a["id"] != agent_id]
        others.append({"id": self.id, "position": self.position, "level": self.level})

        best_fruit, best_prob = None, -1.0
        for fruit in feasible_fruits:
            fruit_level, focal_features, others_features = _build_nn_input(
                np.array(focal_info["position"]), focal_info["level"],
                others, fruit, self.path_finding_grid,
                self._max_agent_level, self._max_fruit_level, self._max_distance,
            )
            fl_t = torch.tensor(fruit_level, dtype=torch.float32).unsqueeze(0)
            fc_t = torch.tensor(focal_features, dtype=torch.float32).unsqueeze(0)
            ot_t = torch.tensor(others_features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = float(self.neural_network(fl_t, fc_t, ot_t)[0, 0])
            self.predictions.append({
                "round": self.round_counter,
                "agent_id": agent_id,
                "trainings_data": (fruit_level, focal_features, others_features),
                "prediction": prob,
                "ground_truth": None,
                "fruit_pos": fruit.position.copy(),
            })
            if prob > best_prob:
                best_prob, best_fruit = prob, fruit
        return best_fruit


    def compute_predicted_path(self, agent_id: int, agent_position: np.ndarray, target_fruit: "Fruit") -> np.ndarray:
        """Compute and store the predicted A* path for the given agent to target_fruit.

        Uses the calling agent's path_finding_grid as an approximation of the other agent's grid.
        Stores the result in predicted_paths, predicted_targets, and prediction_round.

        Args:
            agent_id (int): ID of the agent being predicted
            agent_position (np.ndarray): current position of the predicted agent
            target_fruit (Fruit): the fruit predicted to be the agent's target

        Returns:
            np.ndarray: the predicted path as an array of positions
        """
        closest_slot = min(
            target_fruit.free_slots,
            key=lambda slot: np.linalg.norm(slot - agent_position),
        )
        path = self.get_path(agent_position, closest_slot)
        if path is None:
            return None
        self.predicted_paths[agent_id] = path
        self.predicted_targets[agent_id] = target_fruit
        self.prediction_round[agent_id] = self.round_counter
        return path


    def _get_latest_prediction_prob(self, agent_id: int, fruit: "Fruit") -> float:
        """Return the most recent prediction probability for agent_id targeting fruit.

        Scans self.predictions in reverse (most recent first) so the latest round's
        prediction takes precedence.

        Args:
            agent_id (int): the agent whose prediction to look up
            fruit (Fruit): the fruit to look up

        Returns:
            float: probability, or 0.0 if no prediction found
        """
        for prediction in reversed(self.predictions):
            if (prediction["agent_id"] == agent_id
                    and np.array_equal(prediction["fruit_pos"], fruit.position)):
                return prediction["prediction"]
        return 0.0


    def _compute_threshold(self, current_round_predictions: list, num_feasible_fruits: int) -> float:
        """Compute dynamic threshold as max(Q1 of current-round predictions, 1/num_fruits).

        Q1 (25th percentile) filters out agents unlikely to target a fruit.
        The 1/num_fruits floor ensures the threshold is at least as selective as random.

        Args:
            current_round_predictions (list[float]): NN probabilities from this decision cycle
            num_feasible_fruits (int): number of fruits the agent can consider

        Returns:
            float: the threshold value
        """
        random_baseline = 1.0 / max(num_feasible_fruits, 1)
        if not current_round_predictions:
            return random_baseline
        q1 = float(np.percentile(current_round_predictions, 25))
        return max(q1, random_baseline)


    def select_fruit_by_expected_reward(self, feasible_fruits: list) -> "Fruit":
        """Select the fruit maximizing expected reward using combinatorial agent probabilities.

        For solo-loadable fruits: E[R] = agent_level * fruit_level.
        For cooperative fruits: enumerate all subsets of filtered candidate helpers and compute
        E[R] = sum over subsets S of P(S) * R(S), where P(S) = product of probabilities.

        Dynamic threshold filters low-probability helpers before subset enumeration.
        No distance penalty — the NN already encodes distance in its predictions.

        Args:
            feasible_fruits (list[Fruit]): fruits the agent can feasibly target

        Returns:
            Fruit: the fruit with the highest expected reward, or None if list is empty
        """
        if not feasible_fruits:
            return None

        current_round_probs = [
            p["prediction"]
            for p in self.predictions
            if p["round"] == self.round_counter and p["agent_id"] == self.id
        ]
        threshold = self._compute_threshold(current_round_probs, len(feasible_fruits))

        best_fruit = None
        best_expected_reward = -1.0

        for fruit in feasible_fruits:
            if self.level >= fruit.level:
                expected_reward = float(self.level * fruit.level)
            else:
                candidate_helpers = []
                for agent_info in self.known_agents:
                    prob = self._get_latest_prediction_prob(agent_info["id"], fruit)
                    if prob > threshold:
                        candidate_helpers.append({
                            "level": agent_info["level"],
                            "prob": prob,
                        })

                max_helpers = max(len(fruit.free_slots) - 1, 0)
                candidate_helpers.sort(key=lambda x: x["prob"], reverse=True)
                candidate_helpers = candidate_helpers[:max_helpers]

                expected_reward = 0.0
                n = len(candidate_helpers)
                for subset_mask in range(1 << n):
                    subset = [candidate_helpers[i] for i in range(n) if subset_mask & (1 << i)]
                    not_in_subset = [candidate_helpers[i] for i in range(n) if not (subset_mask & (1 << i))]

                    prob_subset = (
                        np.prod([h["prob"] for h in subset]) if subset else 1.0
                    ) * (
                        np.prod([1.0 - h["prob"] for h in not_in_subset]) if not_in_subset else 1.0
                    )

                    level_sum = self.level + sum(h["level"] for h in subset)
                    reward = float(self.level * fruit.level) if level_sum >= fruit.level else 0.0
                    expected_reward += prob_subset * reward

            if expected_reward > best_expected_reward:
                best_expected_reward = expected_reward
                best_fruit = fruit

        return best_fruit


    def learn(self) -> list[float]:
        """Train the neural network on all predictions that have a ground truth label.

        For each labeled prediction, calls model.fit to perform a forward + backward pass
        (even if a prediction was made earlier, weights may have changed).
        Removes trained predictions from self.predictions afterwards.

        Returns:
            list[float]: MSE loss value for each trained prediction, or empty list if no NN.
        """
        if self.neural_network is None:
            return []
        loss_fn = nn.MSELoss()
        labeled = [p for p in self.predictions if p["ground_truth"] is not None]
        losses: list[float] = []
        for prediction in labeled:
            fruit_level, focal_features, others_features = prediction["trainings_data"]
            fl_t = torch.tensor(fruit_level, dtype=torch.float32).unsqueeze(0)
            fc_t = torch.tensor(focal_features, dtype=torch.float32).unsqueeze(0)
            ot_t = torch.tensor(others_features, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([[prediction["ground_truth"]]], dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.neural_network(fl_t, fc_t, ot_t)
            loss = loss_fn(output, target_tensor)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
        self.predictions = [p for p in self.predictions if p["ground_truth"] is None]
        return losses


    def init_neural_network(self) -> None:
        """Initialize the attention-based AgentPredictor neural network.

        Creates an AgentPredictor with shared agent encoding, attention pooling,
        and a decision network. Architecture is permutation-invariant over other agents.

        Returns:
            None (sets self.neural_network and self.optimizer side effects)
        """
        from neuroevolution import AgentPredictor
        model = AgentPredictor(
            embedding_dim=self.embedding_dim,
            decision_hidden=self.decision_hidden,
        )
        self.neural_network = model
        self.optimizer = torch.optim.Adam(model.parameters())
        
    
    
    def get_possible_coop_level_sums(self, other_agent_levels: list[int]) -> np.ndarray:
        """Compute all feasible level sums for solo and cooperative loading.

        Given the agent's own level and the levels of other agents, enumerates all possible
        combinations (duo, trio, squad) and returns sorted unique sums including solo level.
        Max 4 agents can cooperate (1 + up to 3 others).

        Args:
            other_agent_levels: levels of the other agents (not including self)

        Returns:
            Sorted array of unique feasible level sums (self.level + 0/more others)
        """
        all_levels = [self.level] + other_agent_levels

        duo_coop_levels = pd.Series([np.sum(combo) for combo in itertools.combinations(all_levels, 2)]).unique().tolist()
        triple_coop_levels = pd.Series([np.sum(combo) for combo in itertools.combinations(all_levels, 3)]).unique().tolist()
        squad_coop_levels = pd.Series([np.sum(combo) for combo in itertools.combinations(all_levels, 4)]).unique().tolist()

        res = np.sort(pd.Series([self.level] + duo_coop_levels + triple_coop_levels + squad_coop_levels).unique())

        return res
    
    
    
    
    """Pathfinding to chosen target"""
    
    def get_path(self, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
        """Compute an A* path from start to end position.

        Uses the instance's path_finding_grid (1=walkable, 0=obstacle) and returns
        the path as an array of (row, col) positions including start and end.

        Args:
            start: starting (row, col) position
            end: target (row, col) position

        Returns:
            Array of (row, col) positions along the path, or None if no path exists
        """
        # define the grid
        grid = Grid(matrix=self.path_finding_grid)
        # find the path to the target
        # numpy goes row, column, pathfinding goes column, row. Just... why? What kind of madness is this?
        path, runs = self.pathfinding_alg.find_path(grid.node(start[1], start[0]), 
                                                    grid.node(end[1], end[0]), 
                                                    grid)
        # define the current path of the agent as a list of arrays
        path = np.array([[node.y, node.x] for node in path])
        if len(path) == 0:
            return None
        return path
    
    

    """ Utils for the agent """

    def get_direction_from_other_agent(self, position_history: list[np.array]) -> list[str]:
        """Get the next direction of another agent from the position history of the agent.
        
        Args:
            position_history (list[np.array]): the position history of the other agent
            
        Returns:
            list[str]: list of two possible directions e.g. ["north", "west"]
        """
        
        last_remembered_position = position_history[-self.memory_size:][0]
        current_position = position_history[-1]
        pos_diff = last_remembered_position - current_position # assumes that the current position was the target to get the next possible directions
        return self.get_next_possible_directions(pos_diff)
    
    
    
    def get_next_possible_directions(self, positional_difference: np.ndarray) -> list[str]:
        """Infer direction(s) from a positional difference vector.

        Given a (drow, dcol) difference, returns the cardinal direction(s) of movement.
        Always returns exactly two directions (one for row, one for column).

        Args:
            positional_difference: (drow, dcol) position difference

        Returns:
            List of two cardinal directions, e.g. ["north", "west"]
        """
        
        possible_directions = []
        # check row for possible movement
        if positional_difference[0] > 0:
            possible_directions.append("north")
        else:
            possible_directions.append("south")
        # check column for possible movement
        if positional_difference[1] > 0:
            possible_directions.append("west")
        else:
            possible_directions.append("east")
            
        assert possible_directions and len(possible_directions) == 2, "Error in possible directions"
        return possible_directions
    
    #TODO
    # add a function to get from the directions from get_direction_from_other_agent to the fruits in that direction. These fruits then get a dummy 
    # telling the neural network that the fruit is in the direction the agent is currently runnign to.
    
    
    
    
    
    
    
    
    
    
    
    
