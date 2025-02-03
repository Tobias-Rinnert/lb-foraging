import numpy as np
import random
from dataclasses import dataclass

class Agent():
    """
    Class to save the information of the agent and handle its cognition.
    """
    
    def __init__(self, id: int, position: np.array, level: int, current_target: np.array = None):
        self.current_target = current_target
        self.position = position
        self.level = level
        self.id = id

    
    def __repr__(self):
        return f"Agent {self.id} at position {self.position} with level {self.level} and current target {self.current_target}"
    

    """Target selection"""
    
    def choose_fruit(self, fruits:list): 
            pos = self.position
            level = self.level
            
            # just for testing without nn. If the agent has no current target or 
            # the current target is no longer on the map in the fruit infos, choose a new target
            fruit_positions = np.array([fruit.position for fruit in fruits])
            current_target_still_in_game = np.any(np.all(self.current_target == fruit_positions, axis=1))
            if (self.current_target is None or not current_target_still_in_game):
                # choose a new target fruit     
                feasible_fruits_position = [fruit.position for fruit in fruits if fruit.level <= level]
                self.current_target = random.choice(feasible_fruits_position)  
    
    
    
    """Pathfinding to chosen target"""
    
    def get_next_possible_directions(self, positional_difference) -> list[str]:
            """Get the possible directions given the positional difference. This is an extra step before get_move_differential for readability.

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
        
        
    def get_step(self, direction: str) -> np.array:
            """Translate the direction into a move differential in terms of a numpy array

            Args:
                direction (str): north, south, west, east

            Returns:
                np.array: returns a numpy array with the move differential e.g. [-1, 0] (up) for north
            """
            match direction:
                case "north":
                    return np.array([-1, 0])
                case "south":
                    return np.array([1, 0])
                case "west":
                    return np.array([0, -1])
                case "east":
                    return np.array([0, 1])
                case _:
                    return np.array([0, 0])
                
                
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
    

    def get_next_pos_values(self, possible_directions: list[str], full_info_field: np.array) -> np.array:
        """Get the values of the next positions given the possible directions

        Args:
            full_info_field (np.array): full info field from lbf_gym
            possible_directions (list[str]): list of two possible directions e.g. ["north", "west"]

        Returns:
            np.array: returns the values of the next positions given the possible directions
        """
        next_pos_values = []
        for direction in possible_directions:
            # get the array to addd to the current the position to get the next position given the direction
            step = self.get_step(direction)
            full_field_value_next_pos =  full_info_field[*self.position + step]
            next_pos_values.append(full_field_value_next_pos)
        return np.array(next_pos_values)
    
    
    def choose_next_action_of_player(self, full_info_field) -> np.int64:
        """
        After a target fruit is chosen, choose the next action of the player 
        given the player position and the positional difference to the fruit.
        When a fruit is in the way the player will go around it. 
        Other players are ignored. Possible collisions between players is handled by the enviroment.
        This algorithm only works for the simple scenario of no larger objects in the way of teh agent. 
        In case of larger objects spanning multiple cells use A* or other pathfinding algorithms.
        
        Args:
            full_info_field (np.array): full info field from lbf_gym

        Returns:
            str: the next action of the player one of "north", "south", "west", "east", "load"
        """
        
        #TODO: add a logic to move out of the way off players. 
        # Must be handles so that if an agent stands before a fruit and another agents wants to also get to teh fruit, 
        # the agent that arrives chooses a free spot next to the fruit 
        # And try to change code in env so that when both player try to enter one cell, one wins. Should stop agents getting stuck
        # Maybe choose A* after all since it would solve almost all problems. 
        
        # calculate the positional difference between the player and the target fruit
        positional_difference = self.position - self.current_target
        
        # check if the player stands next to the target fruit to load it 
        if (np.all(np.abs(positional_difference) == np.array([1,0])) or 
            np.all(np.abs(positional_difference) == np.array([0,1]))):
            next_action = "load"
            return self.action_string_to_int(next_action)
        
        # if not choose the next direction to walk to
        possible_directions = self.get_next_possible_directions(positional_difference)
        # get the values of the next positions given the possible directions    
        next_pos_values = self.get_next_pos_values(possible_directions = possible_directions, full_info_field = full_info_field)
                
        if np.all(next_pos_values <= np.array([0,0])):
            # if for both possible directions the next field is free, choose the direction (axis) in which the distance to the fruit is greater
            next_action = possible_directions[np.argmax(np.abs(positional_difference))]
            return self.action_string_to_int(next_action)

        elif np.all(next_pos_values > 0):
            # if both next values are fruits, they are both not wanted. Get the two other directions and check for them if they are free. 
            possible_directions = [direction for direction in ["north", "south", "west", "east"] if direction not in possible_directions]
            next_pos_values = self.get_next_pos_values(possible_directions)
            if np.all(next_pos_values <= np.array([0,0])):
                # if for both possible directions the next field is free, choose the direction (axis) in which the distance to teh fruit is less, 
                # making the behaviour more realistic
                next_action = possible_directions[np.argmin(np.abs(positional_difference))]
                return self.action_string_to_int(next_action)
        
        # if only one of the next fields is free, choose that direction
        next_action = possible_directions[np.where(next_pos_values <= 0)[0][0]]  
        
        return self.action_string_to_int(next_action)