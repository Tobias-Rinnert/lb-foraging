import time
import numpy as np
from dataclasses import dataclass

from .agent import Agent


@dataclass
class Fruit:
    """
    Information of teh fruits in the game
    """
    position: np.array
    level: int
    

class Lbf_Gym(Agent, Fruit):   
    """
    Class to handle the observation from the lbf environment and train the agents. Inherits from Agent and Fruit classes.
    """
    
    full_info_field: np.array
    fruits: list[Fruit]
    agents: list[Agent]
    
    
    def __init__(self, observation: dict):
        # get the full info field
        self.get_full_info_field(observation)
        # get the player infos
        self.get_agent_infos(observation["player_infos"])
        # get the posiitions and level of the fruit
        self.get_fruit_infos()        
    
    
    def update_observation(self, observation: dict):
        # get the full info field
        self.get_full_info_field(observation)
        # get the posiitions and level of the fruit
        self.get_fruit_infos()
        # get the player infos
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
        fruit_pos = np.where(self.full_info_field > 0)
        fruit_pos = list(zip(fruit_pos[0], fruit_pos[1]))
        fruit_pos = [np.array(pos) for pos in fruit_pos]
        fruits = []
        for fruit_pos in fruit_pos:
            fruit = Fruit(position=fruit_pos, level=self.full_info_field[*fruit_pos])
            fruits.append(fruit)
        self.fruits =  fruits
    
    
    def get_agent_infos(self, agent_infos: dict) -> list[Agent]:
        """Initialize the position and level of the player and add the current target

        Args:
            agent_infos (dict): player infos from the observatio from lbf if full_info_mode is True

        Returns:
            list[Agent]: list of class Agent with the position, level and current target of the players
        """
        agents = []
        for agent in agent_infos:
            agent = Agent(id=agent["id"],
                          position=np.array(agent["position"]), 
                          level=agent["level"])
            agents.append(agent)
        self.agents =  agents
    
    
    def update_agents(self, new_player_infos:dict):
        """Update the position and level of the player

        Args:
            new_player_infos (dict): new player infos from the observatio from lbf if full_info_mode is True
        """

        for new_player_info in new_player_infos:            
            id = new_player_info["id"]
            agent = [agent for agent in self.agents if agent.id == id][0]
            agent.position = np.array(new_player_info["position"])
            agent.level = new_player_info["level"]
            
            
    
    
    
