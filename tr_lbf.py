import lbforaging
import gymnasium as gym
from lbforaging.agents import H1
import time
import numpy as np

class tr_marla:   
    # TODO: create description 
    full_info_field: np.array
    fruit_infos: list[tuple]
    player_infos: list[dict]
    
    
    def __init__(self, observation: dict):
        # get the full info field
        self.full_info_field = self.get_full_info_field(observation)
        # get the player infos
        self.player_infos = self.get_player_infos(observation["player_infos"])
        # get the posiitions and level of the fruit
        self.fruit_infos = self.get_fruit_infos()        
    
    
    def update_observation(self, observation: dict):
        # get the full info field
        self.full_info_field = self.get_full_info_field(observation)
        # get the posiitions and level of the fruit
        self.fruit_infos = self.get_fruit_infos()
        # get the player infos
        self.player_infos = self.update_player_infos(observation["player_infos"])
    
    
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
        return field
    
    
    def get_fruit_infos(self) -> dict:
        """get the position and level of the fruits in the get_full_info_field

        Returns:
            dict: dictionary with the position and level of the fruits
        """
        fruit_pos = np.where(self.full_info_field > 0)
        fruit_pos = list(zip(fruit_pos[0], fruit_pos[1]))
        fruit_pos = [np.array(pos) for pos in fruit_pos]
        fruit_infos = []
        for fruit_pos in fruit_pos:
            level = self.full_info_field[*fruit_pos]
            fruit_infos.append({"position": fruit_pos, "level": level})
        return fruit_infos
    
    
    def get_player_infos(self, player_infos:dict) -> list[dict]:
        """Initialize the position and level of the player and add the current target

        Args:
            player_infos (dict): player infos from the observatio from lbf if full_info_mode is True
            current_target ([type], optional): the current target of the player. Defaults to None.

        Returns:
            list[dict]: list of dictionaries with the position, level and current target of the players
        """
        for player in player_infos:
            player["position"] = np.array(player["position"])
            player["current_target"] = None
        return player_infos
    
    
    def update_player_infos(self, new_player_infos:dict):
        """Update the position and level of the player

        Args:
            new_player_infos (dict): new player infos from the observatio from lbf if full_info_mode is True
        """
        player_infos = []
        for old_player_info, new_player_info in zip(self.player_infos, new_player_infos):
            # this is realy ugly but since i want to alter the original lbf code as little as possible i have to do it this way
            # to know which player in the new obs is which player from the old obs
            # TODO when creating a player class, make two var for the old pos, level and two for the new ones 
            # and also two for the old and new target, so that one can always see the old and new values when debugging
            player_infos.append({"position": np.array(new_player_info["position"]), 
                                 "level": new_player_info["level"], 
                                 "current_target": old_player_info["current_target"]})
        return player_infos
            
            
    def choose_fruit(self, agent) -> int: # TODO edit when creating agent class and put into agent class. Then the ugly dict logic falls away
        # pos = agent["position"]
        # level = agent["level"]
        
        # just for testing without nn. If the agent has no current target or 
        # the current target is no longer on the map in the fruit infos, choose a new target
        if (not agent["current_target"] or 
            agent["current_target"] not in self.fruit_infos):
            agent["current_target"] = self.fruit_infos[np.random.randint(0, len(self.fruit_infos))]
        
        return agent["current_target"]
            
        
    def get_next_possible_directions(self, pos_diff:np.array) -> list[str]:
        """Get the possible directions given the positional difference. This is an extra step before get_move_differential for readability.

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
    
    
    def action_string_to_int(self, action: str):
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
    
    
    def choose_next_action_of_player(self, player_position: np.array, target_position: np.array) -> str:
        """
        After a target fruit is chosen, choose the next action of the player 
        given the player position and the positional difference to the fruit.
        When a fruit is in the way the player will go around it. 
        Other players are ignored. Possible collisions between players is handled by the enviroment.
        This algorithm only works for the simple scenario of no larger objects in the way of teh agent. 
        In case of larger objects spanning multiple cells use A* or other pathfinding algorithms.

        Args:
            player_position (np.array): the position of the player on the field
            positional_difference (np.array): player position - fruit position

        Returns:
            str: the next action of the player one of "north", "south", "west", "east", "load"
        """
        
        # calculate the positional difference between the player and the target fruit
        positional_difference = player_position - target_position
        
        # check if the player stands next to the target fruit to load it 
        if (np.all(np.abs(positional_difference) == np.array([1,0])) or 
            np.all(np.abs(positional_difference) == np.array([0,1]))):
            next_action = "load"
            return self.action_string_to_int(next_action)
        
        # if not choose the next direction to walk to
        possible_directions = self.get_next_possible_directions(positional_difference)
        # get the values of the next positions given the possible directions    
        next_pos_values = self.get_next_pos_values(player_position, possible_directions)
                
        if np.all(next_pos_values <= np.array([0,0])):
            # if for both possible directions the next field is free, choose the direction (axis) in which the distance to the fruit is greater
            next_action = possible_directions[np.argmax(np.abs(positional_difference))]
            return self.action_string_to_int(next_action)

        elif np.all(next_pos_values > 0):
            # if both next values are fruits, they are both not wanted. Get the two other directions and check for them if they are free. 
            possible_directions = [direction for direction in ["north", "south", "west", "east"] if direction not in possible_directions]
            next_pos_values = self.get_next_pos_values(player_position, possible_directions)
            if np.all(next_pos_values <= np.array([0,0])):
                # if for both possible directions the next field is free, choose the direction (axis) in which the distance to teh fruit is less, 
                # making the behaviour more realistic
                next_action = possible_directions[np.argmin(np.abs(positional_difference))]
                return self.action_string_to_int(next_action)
    	
        # if only one of the next fields is free, choose that direction
        next_action = possible_directions[np.where(next_pos_values <= 0)[0][0]]  
        
        return self.action_string_to_int(next_action)
    
    
