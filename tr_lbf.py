import lbforaging
import gymnasium as gym

from lbforaging.agents.heuristic_agent import H1



# register the environment in gym

field_size = 8 # size of the game board
number_players = 2 # Number of players
max_num_food = 63 # max amount of food on the board. TODO How is teh amount of food determined?
coop_mode = False # If true, all foods will have teh max level so that all foods can only be loaded by working with other players
max_episode_steps = 50 # Number of steps until one round (episode) is terminated
sight = field_size #  How far can the agents see i presume TODO
max_player_level = 2
min_player_level = 2
max_food_level = 4
min_food_level = 4

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
        },
    )


# define the environment. A more detailed way is discribed on https://github.com/semitable/lb-foraging
env = gym.make(id_string) # "Foraging-{GRID_SIZE}x{GRID_SIZE}-{PLAYER COUNT}p-{FOOD LOCATIONS}f{-coop IF COOPERATIVE MODE}-v0"
# render_mode is "human" per default

# reset the environment with a seed
observation, info = env.reset(seed=42)

episode_over = False
step_amount = 0
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    #use a heuristic to move
    action = H1.step(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    for i in range(50):
        env.render() # TODO: work with timestart and timeend here to get actual frame rate
    episode_over = terminated or truncated
    step_amount += 1
env.close()
print(f"Game lasted {step_amount} steps")