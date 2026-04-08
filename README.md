# TR LBF Foraging

RL agents for the Level-Based Foraging environment, exploring n-order belief estimation for cooperative multi-agent decision making.

## Overview

This project builds on the [Level-Based Foraging (LBF)](https://github.com/semitable/lb-foraging) environment. The idea is to automate walking and loading with A* pathfinding so that the only decision to make is which fruit to target. This decision depends on first-order beliefs about the chosen fruit of the other agents. A neural network estimates these first-order beliefs.

Predictions are made once per agent and reused until that agent deviates from its predicted A* path. This conditional re-prediction strategy reduces NN calls significantly. Target selection uses combinatorial expected reward over all subsets of likely helpers.

Key goals:
- Use the simplest and most efficient models — automate walking so the neural network can focus on estimating the actual stochastic process
- Investigate whether Q-values are still necessary in this setup — it might suffice to estimate which fruit is going to be loaded by which player, without explicitly modelling time

## Repository Layout

This repo contains **three distinct pieces**. Everything I wrote lives in `tr_lbf_addon/` and `web/`. The `lbforaging/` folder is a vendored copy of the upstream environment and should be treated as a third-party dependency.

- **`tr_lbf_addon/`** — my Python package. Agent/fruit logic, A* pathfinding, neural network, training manager, and the game loop consumed by the web backend.
- **`web/`** — my browser-based GUI for running and watching the agents. React + Vite frontend, Python WebSocket backend.
- **`lbforaging/`** — vendored fork of [semitable/lb-foraging](https://github.com/semitable/lb-foraging) with a handful of tweaks (see *Changes to LBF* below). Its original docs, license, and tests live inside that folder and are otherwise untouched.

```
tr_lbf_addon/           My Python package
  lbf_elements.py       Fruit and Agent classes — decision making, pathfinding, NN,
                        conditional re-prediction, combinatorial expected reward selection
  lbf_gym.py            Training manager — processes observations, updates agents and fruits,
                        records ground truth labels, triggers NN training when fruits are loaded
  game_runner.py        Game loop used by the web backend
  tests/                Unit tests

web/                    Browser GUI
  backend/              Python WebSocket server (server.py)
  frontend/             React + Vite frontend

lbforaging/             Vendored upstream LBF environment (modified fork)
start.bat               One-click launcher for the web app (Windows)
```

## Changes to LBF

The `lbforaging/` directory is a slightly modified fork of the original environment:
- Added agent IDs for tracking
- Added `full_info_mode` to get all information from the observation conveniently
- Changed collision logic: when two agents want the same cell, one fails randomly while the other succeeds

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

# Install frontend dependencies
cd web/frontend
npm install
```

## Running the Web App

The web app is the main way to interact with the agents — it runs the `tr_lbf_addon` game loop through a WebSocket backend and renders the board in the browser.

**Windows one-click:**
```
start.bat
```
This launches the backend with auto-reload, the Vite dev server, and opens the browser at `http://localhost:5173`.

**Manual (any OS):**
```sh
# Terminal 1 — backend
python -m uvicorn web.backend.server:app --reload --port 8000

# Terminal 2 — frontend
cd web/frontend
npm run dev
```
Then open `http://localhost:5173`.

**Controls:**
- **Play / Pause** — start and stop auto-play
- **Step** — advance one step manually (works while paused)
- **Speed slider** — adjust step delay
- **Settings panel** — configure environment parameters and restart

## Attribution

The LBF environment is by Filippos Christianos et al. See `lbforaging/LICENSE` for the original license.

## TODOs
- add animated plot of learning rate to front end
- make everything run smoothly. test all parameters and check that agents actually learn
- future steps: agents learn during the game now. add generations with mutations in the architecture of the nn to optimize hyper params
- general goal: make sim so that different types of agents develop with different strategies: selfish vs cooperating, and test which parameter combinations lead to which agents
