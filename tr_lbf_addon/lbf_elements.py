import numpy as np
import pandas as pd
import itertools
import random
from dataclasses import dataclass
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from keras import models
from keras import layers



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
        self.last_action: np.int64 = None # the last action of the agent TODO is this needed?
        self.memory_size: int = 3 # the size of the memory of the agent when remembering the last positions of players from their position histories
        self.is_loading: bool = False
        self.path_goal: np.array = None # the current slot of the fruit the agent is targeting
        self.path_finding_grid: np.array = None
        self.current_path: np.array = None
        self.pathfinding_alg = AStarFinder(diagonal_movement=DiagonalMovement.never) # pathfinding algorithm for the agent
        self.neural_network = None # the neural network for the agent to choose a fruit to target
        self.predictions: list[dict] = [] # the predictions of the neural network 

    
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
        TODO: update the description
        
        Returns:
            str: the next action of the player one of 1, 2, 3, 4, 5
        """
        
        # check which of the free slots of the chosen fruit is closest to the current position 
        self.path_goal = min(self.target.free_slots, key=lambda x: np.linalg.norm(x - self.position))
        
        # if the current position is the current_path_goal, the player is going to load
        if np.all(self.position == self.path_goal):
            next_action = "load"
            return self.action_string_to_int(next_action)

        # TODO: if optimization is neccessary introduce confitions when to not update path
        self.current_path = self.get_path(self.position, self.path_goal)
            
        # get the next position in the path given the current position
        next_position = self.current_path[np.where(np.all(self.current_path == self.position, axis=1))[0][0] + 1]
        # get the action to the next position
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
        """
        Choose a fruit to target. The fruit is chosen with a neural network from the fruits the agent knows about.
        """
        # check that the agent has a high enough level to load any fruit
        fruit_levels = [fruit.level for fruit in self.known_fruits]
        assert self.level >= np.min(fruit_levels), f"All fruits are higher level than the agent {self.level}"
        
        # If the agent has no current target or the current target is no longer on the map in the fruit infos, choose a new target
        fruit_positions = np.array([fruit.position for fruit in self.known_fruits])
        if self.target:
            current_target_still_in_game = np.any(np.all(self.target.position == fruit_positions, axis=1))
        if (self.target is None or not current_target_still_in_game):            
            # get all possible level sums for cooperative play
            coop_levels = self.get_possible_coop_level_sums([agent["level"] for agent in self.known_agents])
            # get all fruits loadable by the agent alone or through cooperation and all fruits with an level <= agent level
            feasible_fruits = [fruit for fruit in self.known_fruits if fruit.level in coop_levels or fruit.level <= self.level]
            
            # go through each fruit and predict for each player if the fruit is going to e chosen
            known_agents_id = [agent["id"] for agent in self.known_agents] + [self.id]
            for fruit in feasible_fruits:
                training_data = self.get_training_data(fruit)
                for agent_id in known_agents_id:
                    # prepare input into shape for the neural network
                    training_data.sort_index(inplace=True)
                    agents_info = training_data.loc[agent_id]
                    training_data.drop(id, inplace=True)
                    training_data = np.array([training_data["fruit_level"].iloc[0]] 
                                             + agents_info.tolist() 
                                             + training_data["level"].tolist() 
                                             + training_data["distance_to_fruit"].tolist())
                    training_data = training_data.reshape(1,len(input))
                    # predict if the fruit is going to be chosen
                    y_pred = self.neural_network.predict(training_data)
                    # save the prediction 
                    self.predictions.append({"round": self.round_counter,
                                             "agent_id": agent_id, 
                                             "trainings_data": training_data, 
                                             "prediction": y_pred, 
                                             "ground_truth": None,
                                             "fruit_pos": fruit.position})
                
                
            
            # self.target = random.choice(feasible_fruits)
  
    
    def get_training_data(self, fruit):
        training_data = []
        
        # get distance to fruit of other agents
        for agent in self.known_agents: 
            closest_free_slot = min(fruit.free_slots, key=lambda x: np.linalg.norm(x - agent["position"])) 
            distance_to_fruit = len(self.get_path(agent["position"], closest_free_slot))
            training_data.append({"agent_id": agent["id"], "level": agent["level"],"distance_to_fruit": distance_to_fruit})
        
        # get distance to fruit of agent self
        closest_free_slot = min(fruit.free_slots, key=lambda x: np.linalg.norm(x - self.position))
        distance_to_fruit = len(self.get_path(self.position, closest_free_slot))
        training_data.append({"agent_id": self.id, "level": self.level, "distance_to_fruit": distance_to_fruit})
        
        # create a dataframe from the training data and set the index to the agent id
        training_data = pd.DataFrame(training_data)
        training_data.set_index("id", inplace=True)

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
    
    
    
    def init_neural_network(self) -> models.Sequential:
        """
        Initialize the neural network for the agent to choose a fruit to target.

        Returns:
            models.Sequential: neural network model
        """
        # Create a Sequential model
        model = models.Sequential()
        # input layer size = level and distance to fruit for each player in the game + the level of the fruit
        input_layer_size = (len(self.known_agents) + 1) * 2 + 1 
        model.add(layers.Input(shape=(input_layer_size,)))
        # define the hidden layers
        model.add(layers.Dense(5, activation='relu'))
        # output layer giving the probability for a fruit being chosen
        model.add(layers.Dense(1, activation='sigmoid'))
        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        self.neural_network = model
        
    
    
    def get_possible_coop_level_sums(self, levels):
        """ 
        Get the possible level sums for cooperative play. Max four players can cooperate
        
        Args:
            levels (list[int]): the levels of the agents
        
        """
        duo_coop_levels = pd.Series([np.sum(levels) for levels in list(itertools.combinations(levels, 2))]).unique().tolist()
        tripple_coop_levels = pd.Series([np.sum(levels) for levels in list(itertools.combinations(levels, 3))]).unique().tolist()
        squad_coop_levels = pd.Series([np.sum(levels) for levels in list(itertools.combinations(levels, 4))]).unique().tolist()

        res = np.sort(pd.Series([self.level] + duo_coop_levels + tripple_coop_levels + squad_coop_levels).unique())
        
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
            ValueError("No path found")
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
    
    
    
    
    
    
    
    
    
    
    
    
