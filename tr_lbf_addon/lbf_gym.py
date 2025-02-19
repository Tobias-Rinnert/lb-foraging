import time
import numpy as np
from dataclasses import dataclass

from .lbf_elements import Agent, Fruit

#TODO: unittests

class Lbf_Gym(Agent, Fruit):   
    """
    Class to handle the observation from the lbf environment and train the agents. Inherits from Agent and Fruit classes.
    """
    
    full_info_field: np.array
    fruits: list[Fruit]
    agents: list[Agent]
    
    
    def __init__(self, observation: dict):
        # initialize teh variable agents
        self.agents = None
        # get the full info field
        self.get_full_info_field(observation)
        # get the posiitions and level of the fruit
        self.get_fruit_infos()
        # get the player infos
        self.get_agent_infos(observation["player_infos"])
        
    
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
    
    
    def get_agent_infos(self, agent_infos: dict) -> list[Agent]:
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
            # choose a start fruit
            agent.choose_fruit(self.fruits)
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
            # update the position and write it into the position history
            new_position = np.array(new_player_info["position"])
            agent.position = new_position
            agent.position_history.append(new_position)
            # update the level
            agent.level = new_player_info["level"]
            # update the target fruit
            agent.choose_fruit(self.fruits)

            
            
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
            # get all fields around the agent
            fields_around_agent = np.array([agent.position + np.array([0, 1]),
                                agent.position + np.array([0, -1]),
                                agent.position + np.array([1, 0]),
                                agent.position + np.array([-1, 0])])
            # get all agents around the agent
            agents_around_agent = [other_agent for other_agent in self.agents if other_agent.position.tolist() in fields_around_agent.tolist()]
            for other_agent in agents_around_agent:
                # if the agent is loading, set it as an obstacle
                if other_agent.is_loading:
                    path_finding_grid[*agent.position] = 0
        
        return path_finding_grid
    
    
    def agents_choose_actions(self) -> list[str]:
        """
        Choose the next action for each agent. 
        A simple wrapper around create_path_finding_grid and choose_next_action
        """
        actions = []
        for agent in self.agents:
            # update the path finding grid
            new_path_finding_grid = self.create_path_finding_grid(agent)
            # choose the next action
            action = agent.choose_next_action(new_path_finding_grid)
            actions.append(action)
        return actions
    
            
            
            
    
    
    
