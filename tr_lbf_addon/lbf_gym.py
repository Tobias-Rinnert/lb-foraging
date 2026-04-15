"""LBF game environment management and per-step simulation.

This module implements the LBF_GYM class which manages game state (agents, fruits, grid),
updates observations from the gymnasium environment, coordinates agent actions, and
records ground truth labels for neural network training.
"""

import numpy as np

from lbf_elements import Agent, Fruit


class LBF_GYM(Agent, Fruit):
    """Manages the LBF game state and per-step simulation.

    Tracks agents, fruits, and the game grid. Provides methods to update observations,
    construct pathfinding grids, coordinate agent actions, and record ground truth labels.
    """

    full_info_field: np.ndarray
    "Full game observation array from gymnasium"
    fruits: list[Fruit]
    "List of all fruits currently on the map"
    agents: list[Agent]
    "List of all agents in the game"


    def __init__(self, observation: dict, ca_map: np.ndarray | None = None) -> None:
        """Initialize the LBF game environment from a gymnasium observation.

        Extracts fruits and agents from the observation and prepares internal state.

        Args:
            observation: gymnasium observation dict with keys:
                - "field": full game grid
                - "fruit_infos": list of fruit dicts (position, level)
                - "player_infos": list of agent dicts (id, position, level, etc.)
            ca_map: optional cellular automata terrain map (field_size x field_size),
                0=stone (obstacle), 1=grass. Passed to create_path_finding_grid.
        """
        # initialize the variable agents
        self.agents: list[Agent] | None = None
        # losses captured from last update_agents call; keyed by agent id
        self.last_step_losses_per_agent: dict[int, list[float]] = {}
        self.any_fruit_loaded: bool = False
        "True if at least one fruit was loaded (disappeared) in the most recent observation update"
        "Per-agent NN training losses from the last step"
        # get the full info field
        self.get_full_info_field(observation)
        # get the positions and level of the fruit
        self.get_fruit_infos()
        # get the player infos
        self.initialize_agents(observation["player_infos"], ca_map)


    def update_observation(self, observation: dict, food_growth: dict | None = None,
                           dead_agents: set | None = None,
                           ca_map: np.ndarray | None = None) -> None:
        """Update the observation and record ground truth labels for loaded fruits.

        Saves previous fruits before refreshing so that disappeared fruits can be identified
        and their matching predictions labelled for training.

        Args:
            observation: new observation from the environment
            food_growth: dict mapping tuple(row, col) to float in [0, 1]. Fruits with
                growth < 1.0 are hidden from agents (still growing).
            dead_agents: set of agent IDs that are dead; dead agents skip cognition/learning.
            ca_map: terrain map (0=stone, 1=grass); stone cells become pathfinding obstacles.
        """
        previous_fruits = list(self.fruits) if self.fruits else []
        self.any_fruit_loaded = False  # reset before record_ground_truth may set it
        self.get_full_info_field(observation)
        self.get_fruit_infos(food_growth)
        self.record_ground_truth(previous_fruits)
        self.update_agents(observation["player_infos"], dead_agents, ca_map)


    def get_full_info_field(self, observation: dict) -> None:
        """Extract and store the full game grid from the observation.

        Stores the field where players are negative (−level) and fruits are positive (level).

        Args:
            observation: gymnasium observation dict with "field" key
        """
        field = observation["field"]
        for player in observation["player_infos"]:
            # players are represented as the negative values of their level at their position in the field
            field[player["position"]] = - player["level"]
        self.full_info_field =  field


    def get_fruit_infos(self, food_growth: dict | None = None) -> None:
        """Extract fruit positions and levels from the full info field.

        Identifies positive values (fruits) in the field, computes their free loading slots
        (adjacent walkable positions), and stores as Fruit objects in self.fruits.
        Fruits with food_growth < 1.0 are hidden (still growing) and excluded.

        Args:
            food_growth: dict mapping tuple(row, col) to float in [0, 1]. Fruits not yet
                at 1.0 are skipped (hidden from agents). If None, all fruits are visible.

        Returns:
            None (sets self.fruits side effect)
        """
        fruit_posisitions = np.where(self.full_info_field > 0)
        fruit_posisitions = list(zip(fruit_posisitions[0], fruit_posisitions[1]))
        fruit_posisitions = [np.array(pos) for pos in fruit_posisitions]
        fruits = []
        for fruit_pos in fruit_posisitions:
            if food_growth is not None and food_growth.get(tuple(fruit_pos), 0.0) < 1.0:
                continue  # fruit not yet ripe — hidden from agents
            # get the four fields around the fruit
            load_slots = [fruit_pos + np.array([0, 1]),
                              fruit_pos + np.array([0, -1]),
                              fruit_pos + np.array([1, 0]),
                              fruit_pos + np.array([-1, 0])]
            # get the free slots around the fruit
            free_slots = [slot for slot in load_slots if self.full_info_field[tuple(slot)] == 0]
            # create the fruit
            fruit = Fruit(position=fruit_pos,
                          level=self.full_info_field[*fruit_pos],
                          free_slots=free_slots)

            fruits.append(fruit)
        self.fruits =  fruits


    def initialize_agents(self, agent_infos: list[dict],
                          ca_map: np.ndarray | None = None) -> None:
        """Create Agent objects from initial observation and set up pathfinding grids.

        Args:
            agent_infos: list of agent dicts with keys id, position, level
            ca_map: optional terrain map passed to create_path_finding_grid
        """
        agents = []
        for agent in agent_infos:
            # create the agent
            agent = Agent(id=agent["id"],
                          position=np.array(agent["position"]),
                          level=agent["level"])
            # create the path finding grid
            agent.path_finding_grid = self.create_path_finding_grid(agent, ca_map)
            agents.append(agent)
        self.agents = agents


    def update_agents(self, new_player_infos: list[dict],
                      dead_agents: set | None = None,
                      ca_map: np.ndarray | None = None) -> None:
        """Update agent state, known world, and pathfinding grids.

        For each agent: increments round counter, updates pathfinding grid, passes current
        fruits/agents info. Dead agents only have position/level updated; cognition is skipped.
        Learning is NOT triggered here — it happens in agents_choose_actions() so the
        updated weights are available immediately for the same-step reprediction.

        Args:
            new_player_infos: list of agent dicts with id, position, level, etc.
            dead_agents: set of agent IDs that are dead; they skip cognition.
            ca_map: terrain map passed to create_path_finding_grid.

        Returns:
            None (updates agents as side effect)
        """
        dead_agents = dead_agents or set()

        for new_player_info in new_player_infos:
            # get the agent with the id of the new player info
            id = new_player_info["id"]
            agent = [agent for agent in self.agents if agent.id == id][0]
            # increment the round counter
            agent.round_counter += 1
            # set is_loading based on last step's action (5=LOAD)
            agent.is_loading = bool(agent.last_action == 5)
            # pass the new path_finding_grid to the agent
            agent.path_finding_grid = self.create_path_finding_grid(agent, ca_map)
            # update position and level for all agents (alive and dead)
            new_position = np.array(new_player_info["position"])
            if np.array_equal(new_position, agent.position):
                agent._stationary_steps += 1
            else:
                agent._stationary_steps = 0
            agent.position = new_position
            agent.position_history.append(new_position)
            agent.level = new_player_info["level"]

            if id in dead_agents:
                continue  # skip cognition for dead agents

            # pass information about the fruits and agents to the agent
            agent.known_fruits = self.fruits
            agent.process_agent_infos(self.agents)
            if agent.neural_network is None:
                agent.init_neural_network()



    def create_path_finding_grid(self, agent, ca_map: np.ndarray | None = None) -> np.array:
        """Create a path finding grid where each obstacle is 0. Agents that are loading,
        fruit positions, and stone cells (ca_map==0) are obstacles.

        Args:
            agent (Agent): the agent for which the path finding grid is created
            ca_map: optional terrain map; cells with value 0 (stone) become obstacles.

        Returns:
            np.array: the path finding grid
        """
        # create a grid from the full info field with ones
        path_finding_grid = np.ones_like(self.full_info_field)

        # set any fruit as an obstacle
        for fruit in self.fruits:
            path_finding_grid[*fruit.position] = 0

        if self.agents is not None:
            # Agents that are loading are obstacles — they won't move this step.
            # Walking agents are not marked so A* can plan through them (env handles collisions).
            for other_agent in self.agents:
                if other_agent.id == agent.id:
                    continue
                if other_agent.is_loading:
                    path_finding_grid[tuple(other_agent.position)] = 0

        if ca_map is not None:
            path_finding_grid[ca_map == 0] = 0  # stone cells are impassable

        return path_finding_grid


    def agents_choose_actions(self, fallback_to_closest: bool = True,
                              dead_agents: set | None = None) -> list[np.int64]:
        """Coordinate learning, target selection, and action planning for all agents.

        When a fruit was loaded this step (any_fruit_loaded=True), each alive agent first
        trains its NN on the newly labelled ground-truth predictions, then repredicts with
        the updated weights. This makes the causal chain explicit:
            fruit loaded → record_ground_truth labels predictions
                         → agent.learn() updates NN
                         → choose_fruit(force_reselect=True) repredicts

        Dead agents immediately receive action 0 (no-op).

        Args:
            fallback_to_closest: if True, assign the closest reachable fruit when
                choose_fruit() yields None. If False, the agent stays idle.
            dead_agents: set of agent IDs that are dead; they are skipped entirely.

        Returns:
            List of actions (np.int64) for each agent: 0=none, 1-4=direction, 5=load
        """
        dead_agents = dead_agents or set()
        actions = []
        for agent in self.agents:
            if agent.id in dead_agents:
                agent.last_action = np.int64(0)
                actions.append(np.int64(0))
                continue
            if self.any_fruit_loaded:
                self.last_step_losses_per_agent[agent.id] = agent.learn()
            agent.choose_fruit(force_reselect=self.any_fruit_loaded)
            if agent.target is None and fallback_to_closest:
                agent.target = self._fallback_target(agent)
            if agent.target is None:
                action = np.int64(0)  # no valid target, stay idle
            else:
                action = agent.choose_next_action()
            agent.last_action = action
            actions.append(action)
        return actions


    def _fallback_target(self, agent) -> "Fruit":
        """Return the closest reachable fruit as a fallback target when choose_fruit() yields None.

        Prefers fruits the agent can load solo (level <= agent.level). If none exist, considers
        all fruits with free slots. Returns None if no fruits are available.

        Args:
            agent (Agent): the agent that needs a fallback target

        Returns:
            Fruit or None: the closest reachable fruit
        """
        solo_fruits = [f for f in agent.known_fruits if f.level <= agent.level and f.free_slots]
        candidates = solo_fruits if solo_fruits else [f for f in agent.known_fruits if f.free_slots]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda f: min(np.linalg.norm(slot - agent.position) for slot in f.free_slots),
        )


    def record_ground_truth(self, previous_fruits: list):
        """Label predictions for fruits that disappeared (were loaded) since the last observation.

        For each loaded fruit, scans all agents' predictions matching that fruit position and
        labels them 1.0 if the predicted agent was adjacent to the fruit, 0.0 otherwise.

        Args:
            previous_fruits (list[Fruit]): the fruit list before the current observation update
        """
        if not previous_fruits:
            return

        current_positions = [f.position for f in self.fruits]

        for prev_fruit in previous_fruits:
            fruit_still_present = any(
                np.array_equal(prev_fruit.position, curr_pos) for curr_pos in current_positions
            )
            if fruit_still_present:
                continue

            self.any_fruit_loaded = True

            adjacent_slots = [
                prev_fruit.position + np.array([0, 1]),
                prev_fruit.position + np.array([0, -1]),
                prev_fruit.position + np.array([1, 0]),
                prev_fruit.position + np.array([-1, 0]),
            ]

            for agent in self.agents:
                for prediction in agent.predictions:
                    if prediction["ground_truth"] is not None:
                        continue
                    if not np.array_equal(prediction["fruit_pos"], prev_fruit.position):
                        continue
                    predicted_agent = next(
                        (a for a in self.agents if a.id == prediction["agent_id"]), None
                    )
                    if predicted_agent is None:
                        continue
                    was_adjacent = any(
                        np.array_equal(predicted_agent.position, slot) for slot in adjacent_slots
                    )
                    prediction["ground_truth"] = 1.0 if was_adjacent else 0.0







