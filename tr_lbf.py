import lbforaging
import gymnasium as gym
from lbforaging.agents import H1
import time

class observation_utils:
    
    def get_full_field(observation):
        field = observation["field"]
        for player in observation["player_infos"]:
            field[player["position"]] = - player["level"]
        return field