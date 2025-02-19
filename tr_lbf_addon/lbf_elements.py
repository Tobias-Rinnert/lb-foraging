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
        
        self.target: Fruit = None
        self.position_history: list[np.array] = [] # the hiostory of teh positions of the agent
        self.memory_size: int = 3 # the size of the memory of the agent when remembering the last positions of players from their position histories
        self.is_loading: bool = False
        self.path_goal: np.array = None # the current slot of the fruit the agent is targeting
        self.path_finding_grid: np.array = None
        self.current_path: np.array = None
        self.pathfinding_alg = AStarFinder(diagonal_movement=DiagonalMovement.never) # pathfinding algorithm for the agent

    
    def __repr__(self):
        return f"id: {self.id}, position: {self.position}, level: {self.level}, current target: {self.target.position}, is loading: {self.is_loading}"
    

    """Target selection"""
    
    def choose_fruit(self, fruits:list): 
            pos = self.position
            level = self.level
            
            # just for testing without nn. If the agent has no current target or 
            # the current target is no longer on the map in the fruit infos, choose a new target
            fruit_positions = np.array([fruit.position for fruit in fruits])
            if self.target:
                current_target_still_in_game = np.any(np.all(self.target.position == fruit_positions, axis=1))
            if (self.target is None or not current_target_still_in_game):
                # choose a new target fruit     
                feasible_fruits_position = [fruit for fruit in fruits if fruit.level <= level]
                # TODO: if the fruits are all higher level than the agents, this throws an error
                # TODO: level of other players is also important to cooperate. If a fruit has a level != to a sum of players level it can not be chosen 
                self.target = random.choice(feasible_fruits_position)
  
    
    
    
    """Pathfinding to chosen target"""
        
        
    def positional_difference_to_direction(self, positional_difference: str) -> np.array:
            """Translate an given positional_difference to a direction

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
    
    
    def update_current_path(self, path_finding_grid: np.array) -> np.array:
        """
        Find the path to the target fruit. The path is saved in the current path of the agent.

        Args:
            path_finding_grid (np.array):  array where each obstacle is 0 or negative. Agents around the agent which are loading are also obstacles
        """
        # define the grid
        grid = Grid(matrix=path_finding_grid)
        # find the path to the target
        # numpy goes row, column, pathfinding goes column, row. Just... why? What kind of madness is this?
        path, runs = self.pathfinding_alg.find_path(grid.node(self.position[1], self.position[0]), 
                                                    grid.node(self.path_goal[1], self.path_goal[0]), 
                                                    grid)
        # define the current path of the agent as a list of arrays
        self.current_path = np.array([[node.y, node.x] for node in path])
        if len(path) == 0:
            print("No path found")
    
    
    def choose_next_action(self, new_path_finding_grid: np.array) -> np.int64:
        """
        TODO: update the description
        
        Args:
            path_finding_grid (np.array):  array where each obstacle is 0 or negative. Agents around the agent which are loading are also obstacles

        Returns:
            str: the next action of the player one of "north", "south", "west", "east", "load"
        """
        
        # if the path_finding_grid is not set, set it
        # TODO: Do this in the init of teh gym and in teh gent update
        if self.path_finding_grid is None:
            self.path_finding_grid = new_path_finding_grid
        
        # get the current free slots around the target
        free_slots = self.target.free_slots
        # check which of the free slots is closest to the current position
        closest_free_slot = min(free_slots, key=lambda x: np.linalg.norm(x - self.position))
        
        #TODO check if new fruits in the path have been spawned. If so update the path. 
        # if any fruit next to the current path is gone update the path
        path_relevant_changes =  np.any(new_path_finding_grid != self.path_finding_grid) # has the path_finding_grid changed?
        
        # check if the current path goal has changed. Checks implicetly if the current target has changed 
        # but takes into accoutn that fruits can be right next to each other
        goal_has_changed = np.any(closest_free_slot != self.path_goal)
        
        # if the two conditions are met update the path and the path_goal
        if goal_has_changed or path_relevant_changes:
            self.path_goal = closest_free_slot
            self.update_current_path(new_path_finding_grid)
            
        # if the current position is the current_path_goal, the player is going to loading
        if np.all(self.position == self.path_goal):
            next_action = "load"
            return self.action_string_to_int(next_action)
            
        # get the next position in the path given the current position
        next_position = self.current_path[np.where(np.all(self.current_path == self.position, axis=1))[0][0] + 1]
        # get the positional difference to the next position
        pos_diff = next_position - self.position
        # get the next possible directions
        direction = self.positional_difference_to_direction(pos_diff)
        return self.action_string_to_int(direction)
        

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
    
    #TODO
    # add a function to get from the directions from get_direction_from_other_agent to the fruits in that direction. These fruits then get a dummy 
    # telling the neural network that the fruit is in the direction the agent is currently runnign to.