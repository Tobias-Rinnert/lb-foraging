import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import torch
import torch.nn as nn



@dataclass
class Fruit:
    """
    Information of the fruits in the game.
    
    Args:
        position (np.array): the position of the fruit
        level (int): the level of the fruit
        restricted_zone (list[np.array]): the fields around the fruit
        free_slots (list[np.array]): the free fields around the fruit
    """
    position: np.array
    level: int
    free_slots: list[np.array] = None



class Agent():
    """
    Class to save the information of the agent and handle its cognition.
    """
    
    def __init__(self, id: int, 
                 position: np.array, 
                 level: int):
        
        self.position = position
        self.level = level
        self.id = id
        
        self.round_counter = 0 # the current round of the game
        self.known_fruits: list[Fruit] = None # the fruits the agent knows about
        self.known_agents: list[dict] = None # information about other agents the agent knows about
        self.target: Fruit = None
        self.position_history: list[np.array] = [] # the hiostory of teh positions of the agent
        self.last_action: np.int64 = None # the last action of the agent TODO not needed right now but couold be used later
        self.memory_size: int = 3 # the size of the memory of the agent when remembering the last positions of players from their position histories
        self.is_loading: bool = False
        self.path_goal: np.array = None # the current slot of the fruit the agent is targeting
        self.path_finding_grid: np.array = None
        self.current_path: np.array = None
        self.pathfinding_alg = AStarFinder(diagonal_movement=DiagonalMovement.never) # pathfinding algorithm for the agent
        self.neural_network = None # the neural network for the agent to choose a fruit to target
        self.predictions: list[dict] = [] # the predictions of the neural network
        self.predicted_targets: dict[int, "Fruit"] = {}   # agent_id → predicted fruit
        self.predicted_paths: dict[int, np.ndarray] = {}  # agent_id → predicted A* path
        self.prediction_round: dict[int, int] = {}        # agent_id → round when predicted

    
    def __repr__(self):
        return f"id: {self.id}, position: {self.position}, level: {self.level}, current target: {self.target}, is loading: {self.is_loading}"
    

    """Information processing"""
    
    def process_agent_infos(self, agents):
        """What information can an agent process about another agent.
        
        Args:
            agent (Agent): the agent info from the observation
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
        if not self.target.free_slots:
            return np.int64(0)

        # check which of the free slots of the chosen fruit is closest to the current position
        self.path_goal = min(self.target.free_slots, key=lambda x: np.linalg.norm(x - self.position))

        # if the current position is the current_path_goal, the player is going to load
        if np.all(self.position == self.path_goal):
            return self.action_string_to_int("load")

        self.current_path = self.get_path(self.position, self.path_goal)

        # no path found (blocked by other agents), stay idle
        if self.current_path is None:
            return np.int64(0)

        # get the next position in the path given the current position
        next_position = self.current_path[np.where(np.all(self.current_path == self.position, axis=1))[0][0] + 1]
        pos_diff = next_position - self.position
        direction = self.positional_difference_to_direction(pos_diff)
        return self.action_string_to_int(direction)


    def positional_difference_to_direction(self, positional_difference: str) -> np.array:
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
    
    def choose_fruit(self):
        """Choose a fruit to target using conditional re-prediction with path tracking.

        Predictions persist across timesteps. Re-prediction only occurs when an agent deviates
        from its predicted path, which drastically reduces NN calls. Own target selection uses
        combinatorial expected reward via select_fruit_by_expected_reward().

        Steps:
            1. Validate agent can load at least one fruit.
            2. Skip re-selection if current target is still on the map.
            3. Build feasible fruit list (solo + cooperative).
            4. Invalidate stale predictions for fruits no longer on the map.
            5. For each other agent: skip NN call if still on predicted path, else re-predict.
            6. Select own target by expected reward.
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

        Builds NN input per fruit without mutating the DataFrame (fixes the original mutation bug).
        Stores all predictions in self.predictions. The NN input format matches init_neural_network:
        [fruit_level_norm, focal_level_norm, focal_distance_norm, other_levels..., other_distances...]

        Args:
            agent_id (int): ID of the agent to predict for
            feasible_fruits (list[Fruit]): fruits the agent could feasibly target

        Returns:
            Fruit: the fruit with the highest predicted probability
        """
        if self.neural_network is None:
            return feasible_fruits[0] if feasible_fruits else None

        best_fruit = None
        best_prob = -1.0

        for fruit in feasible_fruits:
            df = self.get_training_data_per_fruit(fruit)
            df_sorted = df.sort_index()
            focal_row = df_sorted.loc[agent_id]
            other_rows = df_sorted.drop(index=agent_id)

            nn_input = np.array(
                [focal_row["fruit_level"]]
                + [focal_row["level"], focal_row["distance_to_fruit"]]
                + other_rows["level"].tolist()
                + other_rows["distance_to_fruit"].tolist()
            ).reshape(1, -1)

            with torch.no_grad():
                input_tensor = torch.tensor(nn_input, dtype=torch.float32)
                y_pred = self.neural_network(input_tensor)
            prob = float(y_pred[0, 0].item())

            self.predictions.append({
                "round": self.round_counter,
                "agent_id": agent_id,
                "trainings_data": nn_input,
                "prediction": prob,
                "ground_truth": None,
                "fruit_pos": fruit.position.copy(),
            })

            if prob > best_prob:
                best_prob = prob
                best_fruit = fruit

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


    def learn(self):
        """Train the neural network on all predictions that have a ground truth label.

        For each labeled prediction, calls model.fit to perform a forward + backward pass
        (even if a prediction was made earlier, weights may have changed).
        Removes trained predictions from self.predictions afterwards.
        """
        if self.neural_network is None:
            return
        loss_fn = nn.MSELoss()
        labeled = [p for p in self.predictions if p["ground_truth"] is not None]
        for prediction in labeled:
            input_tensor = torch.tensor(prediction["trainings_data"], dtype=torch.float32)
            target_tensor = torch.tensor([[prediction["ground_truth"]]], dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.neural_network(input_tensor)
            loss = loss_fn(output, target_tensor)
            loss.backward()
            self.optimizer.step()
        self.predictions = [p for p in self.predictions if p["ground_truth"] is None]


    def get_training_data_per_fruit(self, fruit):
        training_data = []
        
        # get distance to fruit of other agents
        for agent in self.known_agents:
            closest_free_slot = min(fruit.free_slots, key=lambda x: np.linalg.norm(x - agent["position"]))
            path = self.get_path(agent["position"], closest_free_slot)
            distance_to_fruit = len(path) if path is not None else self.path_finding_grid.shape[0] * 2
            training_data.append({"agent_id": agent["id"], "level": agent["level"], "distance_to_fruit": distance_to_fruit})

        # get distance to fruit of agent self
        closest_free_slot = min(fruit.free_slots, key=lambda x: np.linalg.norm(x - self.position))
        path = self.get_path(self.position, closest_free_slot)
        distance_to_fruit = len(path) if path is not None else self.path_finding_grid.shape[0] * 2
        training_data.append({"agent_id": self.id, "level": self.level, "distance_to_fruit": distance_to_fruit})
        
        # create a dataframe from the training data and set the index to the agent id
        training_data = pd.DataFrame(training_data)
        training_data.set_index("agent_id", inplace=True)

        # normalize the levels with min max scaling
        max_level = training_data["level"].max()
        min_level = training_data["level"].min()
        training_data["level"] = (training_data["level"] - min_level) / (max_level - min_level)

        # normalize the distance to the fruit with min max scaling
        max_distance = training_data["distance_to_fruit"].max()
        min_distance = training_data["distance_to_fruit"].min()
        training_data["distance_to_fruit"] = (training_data["distance_to_fruit"] - min_distance) / (max_distance - min_distance)

        # normalize the fruit level with min max scaling
        min_fruit_level = min([fruit.level for fruit in self.known_fruits])
        max_fruit_level = max([fruit.level for fruit in self.known_fruits])
        fruit_level = (fruit.level - min_fruit_level) / (max_fruit_level - min_fruit_level)
        training_data["fruit_level"] = fruit_level
                
        return training_data
    
    
    
    def init_neural_network(self) -> nn.Sequential:
        """
        Initialize the neural network for the agent to choose a fruit to target.

        Returns:
            nn.Sequential: neural network model
        """
        # input layer size = level and distance to fruit for each player in the game + the level of the fruit
        input_layer_size = (len(self.known_agents) + 1) * 2 + 1
        model = nn.Sequential(
            nn.Linear(input_layer_size, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )
        self.neural_network = model
        self.optimizer = torch.optim.Adam(model.parameters())
        
    
    
    def get_possible_coop_level_sums(self, other_agent_levels):
        """
        Get the possible level sums for cooperative play. Max four players can cooperate.

        Args:
            other_agent_levels (list[int]): the levels of the other agents (not including self)

        """
        all_levels = [self.level] + other_agent_levels

        duo_coop_levels = pd.Series([np.sum(combo) for combo in itertools.combinations(all_levels, 2)]).unique().tolist()
        triple_coop_levels = pd.Series([np.sum(combo) for combo in itertools.combinations(all_levels, 3)]).unique().tolist()
        squad_coop_levels = pd.Series([np.sum(combo) for combo in itertools.combinations(all_levels, 4)]).unique().tolist()

        res = np.sort(pd.Series([self.level] + duo_coop_levels + triple_coop_levels + squad_coop_levels).unique())

        return res
    
    
    
    
    """Pathfinding to chosen target"""
    
    def get_path(self, start: np.array, end: np.array) -> np.array:
        """
        Find the path to the target fruit. The path is saved in the current path of the agent.

        Args:
            path_finding_grid (np.array):  array where each obstacle is 0 or negative. Agents around the agent which are loading are also obstacles
            start (np.array): the start position of the agent
            end (np.array): the end position of the agent
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
    
    
    
    def get_next_possible_directions(self, positional_difference) -> list[str]:
        """Get the possible directions given the positional difference. 

        Returns:
            list[str]: list of two possible directions e.g. ["north", "west"]
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
    
    
    
    
    
    
    
    
    
    
    
    
