# TR LBF Foraging

RL agents for the Level-Based Foraging environment, exploring n-order belief estimation for cooperative multi-agent decision making.

## Overview

This project builds on the [Level-Based Foraging (LBF)](https://github.com/semitable/lb-foraging) environment. The idea is to automate walking and loading with A* pathfinding so that the only decision to make is which fruit to target. This decision depends on first-order beliefs about the chosen fruit of the other agents. A neural network estimates these first-order beliefs.

Predictions are made once per agent and reused until that agent deviates from its predicted A* path. This conditional re-prediction strategy reduces NN calls significantly. Target selection uses combinatorial expected reward over all subsets of likely helpers.

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
  lbf_elements.py       Fruit and Agent classes — decision making, pathfinding, neural net,
                        conditional re-prediction, combinatorial expected reward selection
  lbf_gym.py            Training manager — processes observations, updates agents and fruits,
                        records ground truth labels, triggers NN training when fruits are loaded
lbforaging/             Vendored LBF environment (modified fork)
workbench.ipynb         Development notebook and experiments
```

## Mathematical Functions

### Expected Reward (solo)
When `agent_level >= fruit_level`, the agent can load the fruit alone:

```
E[R] = agent_level × fruit_level
```

### Expected Reward (cooperative)
When cooperation is needed, enumerate all subsets S of filtered candidate helpers:

```
E[R] = Σ_S  P(S) × R(S)
```

where:
- `P(S) = Π_{i∈S} P_i × Π_{j∉S} (1 − P_j)` — probability that exactly subset S shows up
- `R(S) = agent_level × fruit_level` if `agent_level + Σ_{i∈S} level_i ≥ fruit_level`, else `0`

No distance penalty is applied — the NN already encodes distance in its inputs.

### Dynamic Threshold
Before evaluating fruits, a threshold filters out low-probability helpers:

```
τ = max(Q1(predictions), 1 / |feasible_fruits|)
```

- `Q1` = 25th percentile of this agent's current-round NN predictions
- `1/|feasible_fruits|` = "better than random" baseline

Only agents with `P > τ` are considered as potential helpers.

### Min-Max Normalisation (NN input)
Applied to agent levels, distances, and fruit levels before feeding the neural network:

```
x_norm = (x − x_min) / (x_max − x_min)
```

### Path Distance
A* shortest path length using cardinal moves only (no diagonals). Used both for navigation and as a feature in the NN input.

### Reward Formula (environment)
```
reward = agent_level × food_level
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
