# TR LBF Foraging

RL agents for the Level-Based Foraging environment, exploring n-order belief estimation for cooperative multi-agent decision making.

## Overview

This project builds on the [Level-Based Foraging (LBF)](https://github.com/semitable/lb-foraging) environment. The idea is to automate walking and loading with A* pathfinding so that the only decision to make is which fruit to target. This decision depends on first-order beliefs about the chosen fruit of the other agents. A neural network estimates these first-order beliefs.

Key goals:
- Use the simplest and most efficient models — automate walking so the neural network can focus on estimating the actual stochastic process
- Investigate whether Q-values are still necessary in this setup — it might suffice to estimate which fruit is going to be loaded by which player, without explicitly modelling time

## Changes to LBF

The `lbforaging/` directory contains a slightly modified fork of the original LBF environment:
- Added agent IDs for tracking
- Added `full_info_mode` to get all information from the observation conveniently
- Changed collision logic: when two agents want the same cell, one fails randomly while the other succeeds

The original LBF documentation, license, and tests are in `lbforaging/`.

## Project Structure

```
tr_lbf_addon/           Main package
  lbf_elements.py       Fruit and Agent classes (decision making, pathfinding, neural net)
  lbf_gym.py            Training manager — processes observations, updates agents and fruits
lbforaging/             Vendored LBF environment (modified fork)
workbench.ipynb         Development notebook and experiments
```

## Installation

```sh
git clone <repo-url>
cd lb-foraging
pip install -e .
```

## Usage

See `workbench.ipynb` for the current development workflow. The notebook creates an LBF environment, initializes the `Lbf_Gym` manager, and runs episodes with agent decision making.

```python
import lbforaging
import gymnasium as gym
from tr_lbf_addon.lbf_gym import Lbf_Gym

env = gym.make("Foraging-8x8-2p-1f-v3")
observation, info = env.reset(seed=42)
manager = Lbf_Gym(observation[0])
```

## Attribution

The LBF environment is by Filippos Christianos et al. See `lbforaging/LICENSE` for the original license.
