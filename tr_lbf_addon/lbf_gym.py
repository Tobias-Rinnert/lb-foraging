import numpy as np

from lbf_elements import Agent, Fruit

#TODO: unittests

class LBF_GYM(Agent, Fruit):   
    """
    Class to handle the observation from the lbf environment and train the agents. Inherits from Agent and Fruit classes.
    """
    
    full_info_field: np.array
    fruits: list[Fruit]
    agents: list[Agent]
    
    
    def __init__(self, observation: dict):
        # initialize teh variable agents
        self.agents = None
        # losses captured from last update_agents call; keyed by agent id
        self.last_step_losses_per_agent: dict[int, list[float]] = {}
        # get the full info field
        self.get_full_info_field(observation)
        # get the posiitions and level of the fruit
        self.get_fruit_infos()
        # get the player infos
        self.initialize_agents(observation["player_infos"])
        
    
    def update_observation(self, observation: dict):
        """Update the observation and record ground truth labels for loaded fruits.

        Saves previous fruits before refreshing so that disappeared fruits can be identified
        and their matching predictions labelled for training.

        Args:
            observation (dict): new observation from the environment
        """
        previous_fruits = list(self.fruits) if self.fruits else []
        self.get_full_info_field(observation)
        self.get_fruit_infos()
        self.record_ground_truth(previous_fruits)
        self.update_agents(observation["player_infos"])
    
    
    def get_full_info_field(self, observation: dict) -> np.array:
        """
        Create a field where players are represented as the negative values of their level,
        and fruit are represented as positive values with their levels.

        Args:
            observation (dict): the observatio from lbf if full_info_mode is True

        Returns:
            np.array: the field with players and fruits
        """
        field = observation["field"]
        for player in observation["player_infos"]:
            # players are represented as the negative values of their level at their position in the field
            field[player["position"]] = - player["level"]
        self.full_info_field =  field
    
    
    def get_fruit_infos(self) -> list[Fruit]:
        """get the position and level of the fruits in the get_full_info_field

        Returns:
            dict: dictionary with the position and level of the fruits
        """
        fruit_posisitions = np.where(self.full_info_field > 0)
        fruit_posisitions = list(zip(fruit_posisitions[0], fruit_posisitions[1]))
        fruit_posisitions = [np.array(pos) for pos in fruit_posisitions]
        fruits = []
        for fruit_pos in fruit_posisitions:
            # get teh four fields around the fruit
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
    
    
    def initialize_agents(self, agent_infos: dict) -> list[Agent]:
        """Initialize the position and level of the player and add the current target

        Args:
            agent_infos (dict): player infos from the observatio from lbf if full_info_mode is True

        Returns:
            list[Agent]: list of class Agent with the position, level and current target of the players
        """
        agents = []
        for agent in agent_infos:
            # create the agent            
            agent = Agent(id=agent["id"],
                          position=np.array(agent["position"]), 
                          level=agent["level"])
            # create the path finding grid
            agent.path_finding_grid = self.create_path_finding_grid(agent)
            agents.append(agent)
        self.agents = agents
    
    
    def update_agents(self, new_player_infos:dict):
        """Update the position and level of the player

        Args:
            new_player_infos (dict): new player infos from the observatio from lbf if full_info_mode is True
        """

        for new_player_info in new_player_infos:  
            # get the agent with the id of the new player info          
            id = new_player_info["id"]
            agent = [agent for agent in self.agents if agent.id == id][0]
            # increment the round counter
            agent.round_counter += 1
            # pass the new path_finding_grid to the agent
            agent.path_finding_grid = self.create_path_finding_grid(agent)
            # pass information about the fruits and agents to the agent
            # TODO if information asymmetries are allowed, this should be changed here
            agent.known_fruits = self.fruits
            agent.process_agent_infos(self.agents)
            if agent.neural_network is None:
                agent.init_neural_network()
            # update the position and write it into the position history
            new_position = np.array(new_player_info["position"])
            agent.position = new_position
            agent.position_history.append(new_position)
            # update the level
            agent.level = new_player_info["level"]
            self.last_step_losses_per_agent[agent.id] = agent.learn()

            
            
    def create_path_finding_grid(self, agent) -> np.array:
        """Create a path finding grid where each obstacle is 0 or negative. Agents around the agent which are loading are also obstacles

        Args:
            agent (Agent): the agent for which the path finding grid is created

        Returns:
            np.array: the path finding grid
        """
        # create a grid from the full info field with ones
        path_finding_grid = np.ones_like(self.full_info_field)

        # set any fruit as an obstacle
        for fruit in self.fruits:
            path_finding_grid[*fruit.position] = 0

        if self.agents is not None:
            for other_agent in self.agents:
                if other_agent.id == agent.id:
                    continue
                # mark other agents as obstacles so we path around them
                path_finding_grid[*other_agent.position] = 0

        return path_finding_grid
    
    
    def agents_choose_actions(self, fallback_to_closest: bool = True) -> list[str]:
        """Choose the next action for each agent.

        Calls choose_fruit() for target selection, then optionally falls back to the closest
        reachable fruit if no target was assigned (e.g. when the NN is not yet initialised).

        Args:
            fallback_to_closest: if True, assign the closest reachable fruit when
                choose_fruit() yields None. If False, the agent stays idle.
        """
        actions = []
        for agent in self.agents:
            agent.choose_fruit()
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
    
            
            
            
    
    
    
