import lbforaging
import gymnasium as gym
from lbforaging.agents import H1
import time
import numpy as np

class tr_marla:   
    # TODO: create description 
    full_info_field: np.array
    fruit_positions: list[tuple]
    player_infos: list[dict]
    
    
    def __init__(self, observation: dict):
        # get the full info field
        self.full_info_field = self.get_full_info_field(observation)
        # get the posiition of the fruit
        self.fruit_positions = self.get_fruit_pos_from_full_field()
        # get the player infos
        self.player_infos = observation["player_infos"]
    
    
    def get_full_info_field(self, observation: dict) -> np.array:
        """
        Create a field where players are represented as the negative values of their level,
        anbd fruit are represented as positive values with their levels.

        Args:
            observation (dict): the observatio from lbf if full_info_mode is True

        Returns:
            np.array: the field with players and fruits
        """
        field = observation["field"]
        for player in observation["player_infos"]:
            # players are represented as the negative values of their level at their position in the field
            field[player["position"]] = - player["level"]
        return field
    
    
    def get_fruit_pos_from_full_field(self) -> list[tuple]:
        """get the position of the fruits in the get_full_info_field

        Args:
            full_field (np.array): output from get_full_info_field()

        Returns:
            list[tuple]: list of tuples with coordinates of the fruits (row, col)
        """
        fruit_pos = np.where(self.full_info_field > 0)
        fruit_pos = list(zip(fruit_pos[0], fruit_pos[1]))
        fruit_pos = [np.array(pos) for pos in fruit_pos]
        return fruit_pos
    
    
    def get_next_possible_directions(self, pos_diff:np.array) -> list[str]:
        """Get the possible directions given the position difference

        Args:
            pos_diff (np.array): player_position - target_position

        Returns:
            list[str]: list of two possible directions e.g. ["north", "west"]
        """
        possible_directions = []
        # check row for possible movement
        if pos_diff[0] > 0:
            possible_directions.append("north")
        else:
            possible_directions.append("south")
        # check column for possible movement
        if pos_diff[1] > 0:
            possible_directions.append("west")
        else:
            possible_directions.append("east")
            
        assert possible_directions and len(possible_directions) == 2, "Error in possible directions"
        return possible_directions
    
    def get_move_differential(self, direction: str) -> np.array:
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
        
        
    def get_next_pos_values(self, player_position: np.array, possible_directions: list[str]) -> np.array:
        """Get the values of the next positions given the possible directions

        Args:
            player_position (np.array): position of the player on the field
            possible_directions (list[str]): list of two possible directions e.g. ["north", "west"]

        Returns:
            np.array: returns the values of the next positions given the possible directions
        """
        next_pos_values = []
        for direction in possible_directions:
            # get the array to addd to the current the position to get the next position given the direction
            move_diff = self.get_move_differential(direction)
            full_field_value_next_pos =  self.full_info_field[*player_position + move_diff]
            next_pos_values.append(full_field_value_next_pos)
        return np.array(next_pos_values)
    
    
    def choose_next_action_of_player(self, player_position: np.array, positional_difference: np.array) -> str:
        """
        After a target fruit is chosen, choose the next action of the player 
        given the player position and the positional difference to the fruit.
        When a fruit is in the way the player will go around it. 
        Othe rplayers are ignored. Possible collisions between players is handled by the enviroment.

        Args:
            player_position (np.array): the position of the player on the field
            positional_difference (np.array): player position - fruit position

        Returns:
            str: the next action of the player one of "north", "south", "west", "east", "load"
        """
        
        next_action = None

        # check if the player can load the fruit
        if np.any(positional_difference == 0):
            next_action = "load"
            return next_action
        
        # if not choose the next direction to walk to
        possible_directions = self.get_next_possible_directions(positional_difference)
        
        # get the values of the next positions given the possible directions    
        next_pos_values = self.get_next_pos_values(player_position, possible_directions)
                
        if np.all(next_pos_values == np.array([0,0])):
            # if for both possible directions the next field is free, choose the direction (axis) in which the distance to the fruit is greater
            next_action = possible_directions[np.argmax(np.abs(positional_difference))]

        elif np.all(next_pos_values > 0):
            # if both next values are fruits, they are both not wanted. Get the two other directions and check for them if they are free. 
            other_directions = [direction for direction in ["north", "south", "west", "east"] if direction not in possible_directions]
            next_pos_values = self.get_next_pos_values(player_position, other_directions)
            if np.all(next_pos_values == np.array([0,0])):
                # if for both possible directions the next field is free, choose the direction (axis) in which the distance to teh fruit is less, 
                # making the behaviour more realistic
                next_action = other_directions[np.argmin(np.abs(positional_difference))]
            elif np.any(next_pos_values == 0):
                # if one of the next fields is free, move towards that direction
                next_action = other_directions[np.where(next_pos_values == 0)[0][0]]

        else:
            # if only one of the next field is free meaning equal to 0 or below, move towards that direction
            next_action = possible_directions[np.where(next_pos_values == 0)[0][0]]  
        
        return next_action