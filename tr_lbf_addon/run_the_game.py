import gymnasium as gym
import time

from lbf_gym import LBF_GYM

# register the environment in gym

field_size = 8 # size of the game board
number_players = 5 # Number of players
max_num_food = 8 # max amount of food on the board. TODO How is teh amount of food determined?
coop_mode = False # If true, all foods will have teh max level so that all foods can only be loaded by working with other players
max_episode_steps = 50 # Number of steps until one round (episode) is terminated
sight = 0  #  How far can the agents see i presume TODO
max_player_level = 1
min_player_level = 1
max_food_level = 1
min_food_level = 1
normalize_reward = True
grid_observation = False
observe_agent_levels = True # If true, the observation will include the level of the agents
penalty = 0.0 # if the player was not the one to load the food, it gets a penalty to its reward
render_mode = "human"
full_info_mode = True

id_string = "Foraging-{0}x{0}-{1}p-{2}f{3}-v3".format(field_size, number_players, max_num_food, "-coop" if coop_mode else "")

gym.envs.registration.register(
    id=id_string,
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": number_players,
        "max_player_level": max_player_level,
        "min_player_level": min_player_level, 
        "max_food_level": max_food_level,
        "min_food_level": min_food_level,
        "field_size": (field_size, field_size),
        "max_num_food": max_num_food,
        "sight": sight,
        "max_episode_steps": max_episode_steps,
        "force_coop": coop_mode,
        "normalize_reward" : normalize_reward,
        "grid_observation" : grid_observation,
        "observe_agent_levels" : observe_agent_levels,
        "penalty" : penalty,
        "render_mode" : render_mode,
        "full_info_mode": full_info_mode
        },
    )


# define the environment. A more detailed way is discribed on https://github.com/semitable/lb-foraging
env = gym.make(id_string) # "Foraging-{GRID_SIZE}x{GRID_SIZE}-{PLAYER COUNT}p-{FOOD LOCATIONS}f{-coop IF COOPERATIVE MODE}-v0"
# render_mode is "human" per default

# reset the environment with a seed
observation, info = env.reset(seed=42)

# initialize the class
tr_marla_class = LBF_GYM(observation[0])

episode_over = False
step_amount = 0
while not episode_over:
    tr_marla_class.update_observation(observation[0])
    actions = tr_marla_class.agents_choose_actions()
    observation, reward, terminated, truncated, info = env.step(tuple(actions))
    # let 2 senconds pass
    env.render()
    time.sleep(0.2)
    episode_over = terminated or truncated
    step_amount += 1
env.close()
print(f"Game lasted {step_amount} steps")