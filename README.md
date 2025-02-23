
<p align="center">
 <img width="350px" src="docs/img/logo.png" align="center" alt="Level Based Foraging (LBF)" />
 <p align="center">A multi-agent reinforcement learning environment</p>
</p>

<!-- TABLE OF CONTENTS -->
<h1> Table of Contents </h1>

- [Tobias Rinnert project]
[About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Installation](#installation)
- [Usage](#usage)
  - [Observation Space](#observation-space)
  - [Action space](#action-space)
  - [Rewards](#rewards)
- [Human Play](#human-play)
- [Please Cite](#please-cite)
- [Contributing](#contributing)
- [Contact](#contact)


<!-- Tobias Rinnert project -->
# Tobias Rinnert project
This is a fork of lbf_foraging in which I want to implement my own RL algorithm to play around with ideas about RL and social simulations. 
The idea is to first automate walking and loading with astar so that the only decision to make is to choose a fruit to walk to and load. This decision is dependent on the first order beliefe about the chosen fruit of the other agents. A neural network will be implemented that estimates these first oder beliefs. One main focus of this project is to use the simplest and most efficient models. Therefore walking is automated so that the neural network can focus on estimating the actual stochastic process. Furthermore I want to find out if Q values are still necessary in this setup. While estimating when a fruit is going to be loaded is still relevant it might not be necessary to include time as a factor in the estimation but rather just which fruit is going to be loaded by which player. 

I have slightly altered the source code of lb-foraging. Mainly to get all the information in a convenient way from the observation but I also added ids for the agents. 
Furthermore I editied the logic for two agents who want to go to the same cell. If two agents want to get to the same cell, than one of them fails randomely while the other succeeds. 

The rest of the project is inside the tr_lbf_addon dir and currently also in the workbench.ipynb where I develop code and ran tests. The tr_lbf_addon fir holds two files: 
- lbf_gym.py holds the training ground for the agents and acts as a manager for taking new input and updating the agents and fruits. 
- lbf_elements.py holds two classes: Fruits and agents. Decision making is handeled by the agent class. 

> [!CAUTION]
> The LBF environment was updated to support the new [Gymnasium](https://gymnasium.farama.org/) interface in replacement of the deprecated `gym=0.21` dependency (many thanks @LukasSchaefer). For backwards compatibility, please see [Gymnasium compatibility documentation](https://gymnasium.farama.org/content/gym_compatibility/) or use version v1.1.1 of the repository. The main changes to the interface are as follows:
> - `obss = env.reset()` --> `obss, info = env.reset()`
> - `obss, rewards, dones, info = env.step(actions)` --> `obss, rewards, done, truncated, info = env.step(actions)`
> - The `done` flag is now given as a single boolean value instead of a list of booleans.
> - You can give the reset function a particular seed with `obss, info = env.reset(seed=42)` to initialise a particular episode.


<!-- ABOUT THE PROJECT -->
# About The Project

This environment is a mixed cooperative-competitive game, which focuses on the coordination of the agents involved. Agents navigate a grid world and collect food by cooperating with other agents if needed.

<p align="center">
 <img width="450px" src="docs/img/lbf.gif" align="center" alt="Level Based Foraging (LBF) illustration" />
</p>

More specifically, agents are placed in the grid world, and each is assigned a level. Food is also randomly scattered, each having a level on its own. Agents can navigate the environment and can attempt to collect food placed next to them. The collection of food is successful only if the sum of the levels of the agents involved in loading is equal to or higher than the level of the food. Finally, agents are awarded points equal to the level of the food they helped collect, divided by their contribution (their level). The figures below show two states of the game, one that requires cooperation, and one more competitive.


While it may appear simple, this is a very challenging environment, requiring the cooperation of multiple agents while being competitive at the same time. In addition, the discount factor also necessitates speed for the maximisation of rewards. Each agent is only awarded points if it participates in the collection of food, and it has to balance between collecting low-levelled food on his own or cooperating in acquiring higher rewards. In situations with three or more agents, highly strategic decisions can be required, involving agents needing to choose with whom to cooperate. Another significant difficulty for RL algorithms is the sparsity of rewards, which causes slower learning.

This is a Python simulator for level based foraging. It is based on OpenAI's RL framework, with modifications for the multi-agent domain. The efficient implementation allows for thousands of simulation steps per second on a single thread, while the rendering capabilities allows humans to visualise agent actions. Our implementation can support different grid sizes or agent/food count. Also, game variants are implemented, such as cooperative mode (agents always need to cooperate) and shared reward (all agents always get the same reward), which is attractive as a credit assignment problem.



<!-- GETTING STARTED -->
# Getting Started

## Installation

Install using pip
```sh
pip install lbforaging
```
Or to ensure that you have the latest version:
```sh
git clone https://github.com/semitable/lb-foraging.git
cd lb-foraging
pip install -e .
```


<!-- USAGE EXAMPLES -->
# Usage

Create environments with the gym framework.
First import
```python
import lbforaging
```

Then create an environment:
```python
env = gym.make("Foraging-8x8-2p-1f-v3")
```

We offer a variety of environments using this template:
```
"Foraging-{GRID_SIZE}x{GRID_SIZE}-{PLAYER COUNT}p-{FOOD LOCATIONS}f{-coop IF COOPERATIVE MODE}-v0"
```

But you can register your own variation using (change parameters as needed):
```python
from gym.envs.registration register

register(
    id="Foraging-{0}x{0}-{1}p-{2}f{3}-v3".format(s, p, f, "-coop" if c else ""),
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": p,
        "max_player_level": 3,
        "field_size": (s, s),
        "max_food": f,
        "sight": s,
        "max_episode_steps": 50,
        "force_coop": c,
    },
)
```

Similarly to Gym, but adapted to multi-agent settings step() function is defined as
```python
nobs, nreward, ndone, ninfo = env.step(actions)
```

Where n-obs, n-rewards, n-done and n-info are LISTS of N items (where N is the number of agents). The i'th element of each list should be assigned to the i'th agent.



## Observation Space

## Action space

actions is a LIST of N INTEGERS (one of each agent) that should be executed in that step. The integers should correspond to the Enum below:

```python
class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5
```
Valid actions can always be sampled like in a gym environment, using:
```python
env.action_space.sample() # [2, 3, 0, 1]
```
Also, ALL actions are valid. If an agent cannot move to a location or load, his action will be replaced with `NONE` automatically.

## Rewards

The rewards are calculated as follows. When one or more agents load a food, the food level is rewarded to the agents weighted with the level of each agent. Then the reward is normalised so that at the end, the sum of the rewards (if all foods have been picked-up) is one. 
If you prefer code:

```python
for a in adj_players: # the players that participated in loading the food
    a.reward = float(a.level * food_level) # higher-leveled agents contribute more and are rewarded more. 
    if self._normalize_reward:
        a.reward = a.reward / float(
            adj_player_level * self._food_spawned
        )  # normalize reward so that the final sum of rewards is one.
```

<!-- HUMAN PLAY SCRIPT -->
# Human Play

We also provide a simple script that allows you to play the environment as a human. This is useful for debugging and understanding the environment dynamics. To play the environment, run the following command:
```sh
python human_play.py --env <env_name>
```
where `<env_name>` is the name of the environment you want to play. For example, to play an LBF task with two agents and one food in a 8x8 grid, run:
```sh
python human_play.py --env Foraging-8x8-2p-1f-v3
```

Within the script, you can control a single agent at the time using the following keys:
- Arrow keys: move current agent up/ down/ left/ right
- L: load food
- K: load food and let agent keep loading (even if agent is swapped)
- SPACE: do nothing
- TAB: change the current agent (rotates through all agents)
- R: reset the environment and start a new episode
- H: show help
- D: display agent info (at every time step)
- ESC: exit


<!-- CITATION -->
# Please Cite
1. The paper that first uses this implementation of Level-based Foraging (LBF) and achieves state-of-the-art results:
```
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Schäfer, Lukas and Albrecht, Stefano V},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
2. A comperative evaluation of cooperative MARL algorithms and includes an introduction to this environment:
```
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
}
```

<!-- CONTRIBUTING -->
# Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->
# Contact

Filippos Christianos - f.christianos@ed.ac.uk

Project Link: [https://github.com/semitable/lb-foraging](https://github.com/semitable/lb-foraging)

