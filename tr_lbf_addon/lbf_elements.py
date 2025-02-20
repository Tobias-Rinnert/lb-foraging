import numpy as np
import random
from dataclasses import dataclass
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder



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
        
        self.known_fruits: list[Fruit] = None # the fruits the agent knows about
        self.known_agents: list[dict] = None # information about other agents the agent knows about
        self.target: Fruit = None
        self.position_history: list[np.array] = [] # the hiostory of teh positions of the agent
        self.last_action: np.int64 = None # the last action of the agent
        self.memory_size: int = 3 # the size of the memory of the agent when remembering the last positions of players from their position histories
        self.is_loading: bool = False
        self.path_goal: np.array = None # the current slot of the fruit the agent is targeting
        self.path_finding_grid: np.array = None
        self.current_path: np.array = None
        self.pathfinding_alg = AStarFinder(diagonal_movement=DiagonalMovement.never) # pathfinding algorithm for the agent

    
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
            pos = self.position
            level = self.level
            
            fruit_distances = self.get_distances_agents_fruits()
            
            # just for testing without nn. If the agent has no current target or 
            # the current target is no longer on the map in the fruit infos, choose a new target
            fruit_positions = np.array([fruit.position for fruit in self.known_fruits])
            if self.target:
                current_target_still_in_game = np.any(np.all(self.target.position == fruit_positions, axis=1))
            if (self.target is None or not current_target_still_in_game):
                # choose a new target fruit     
                feasible_fruits_position = [fruit for fruit in self.known_fruits if fruit.level <= level]
                # TODO: if the fruits are all higher level than the agents, this throws an error
                # TODO: level of other players is also important to cooperate. If a fruit has a level != to a sum of players level it can not be chosen 
                self.target = random.choice(feasible_fruits_position)
  
    
    def get_distances_agents_fruits(self):
        fruit_distances = []
        for agent in self.known_agents: 
            fruit_distances_per_agent = [] 
            for fruit in self.known_fruits:
                closest_free_slot = min(fruit.free_slots, key=lambda x: np.linalg.norm(x - agent["position"]))
                distance_to_fruit = len(self.get_path(agent["position"], closest_free_slot))
                fruit_distances_per_agent.append({"fruit_position": fruit.position, "distance": distance_to_fruit})
            fruit_distances.append({"agent_id": agent["id"], "fruit_distances": fruit_distances_per_agent})
            
        return fruit_distances
    
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
    
    
    
    
    
    
    
    
    
    
    
    
