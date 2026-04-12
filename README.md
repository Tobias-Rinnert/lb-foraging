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

## How a Simulation Step Works

Each game step follows the same sequence for every agent: update state, learn from past predictions, predict other agents, select a target fruit, and walk or load. The sections below walk through this sequence and the math behind each part.

### 1. State Update and Learning

At the start of each step, `update_agents()` refreshes positions, fruits, and pathfinding grids from the new gymnasium observation. Before the agent makes any decisions, it trains on any predictions that received ground truth labels since the last step.

Learning is **online and supervised**, not reinforcement learning. Ground truth comes from observing what actually happens in the game:

- When a fruit disappears (it was loaded), `record_ground_truth` labels all predictions that referenced that fruit: `1.0` if the predicted agent was adjacent (participated in loading), `0.0` otherwise.
- `agent.learn()` trains on all labeled predictions using MSE loss and Adam optimizer, then discards them. Unlabeled predictions (for fruits still on the map) are kept for future labeling.

```
Game step N:
  Agent α predicts: "Will agent β target fruit F?"
  → stores (input_data, prediction=0.7, ground_truth=None)

Game step N+t:
  Fruit F is loaded! Observation shows F disappeared.
  → record_ground_truth checks: was agent β adjacent to F?
  → Yes → ground_truth = 1.0

Game step N+6:
  update_agents → agent.learn()
  → trains on (input_data, target=1.0), MSE loss, backprop
  → discards this labeled prediction
```

### 2. Predicting Other Agents (Neural Network)

Each agent owns an instance of `AgentPredictor` (defined in `neuroevolution.py`), an attention-based model that answers: *"What is the probability that agent X will target fruit Y?"*

For each (agent, fruit) pair, `_build_nn_input` constructs the structured input and the `AgentPredictor` outputs a probability. Predictions persist across timesteps — an agent only re-runs the NN for another agent when that agent deviates from its predicted A* path, which drastically reduces NN calls.

#### Input normalisation

All NN inputs are normalized against fixed game-settings bounds, not against the current observation:

```
level_norm  = level / max_level_from_settings
dist_norm   = A*_distance / (field_size × 2)
```

This keeps inputs stable across episodes regardless of which agents or fruits happen to be present. Distances are A* shortest path lengths using cardinal moves only (no diagonals).

#### Architecture

Refernces:
https://arxiv.org/abs/1703.06114
https://arxiv.org/abs/1706.03762

The number of agents can vary between games. Attention naturally handles variable-length sets while being **permutation-invariant** — swapping two agents in the input produces the same output. All agents are visible in a single forward pass, enabling n-th order belief reasoning.

```
                    ┌─────────────────────────────────┐
                    │         AgentPredictor          │
                    └─────────────────────────────────┘

  Inputs:
    fruit_level      (batch, 1)    — normalized fruit level
    focal_features   (batch, 2)    — [level, distance] of agent being predicted
    others_features  (batch, N, 2) — [level, distance] of every other agent (variable N)

  ┌──────────────────────────────────────────────────────────┐
  │  1. Shared Encoder (φ)                                   │
  │     nn.Linear(2, 8) → ReLU                               │
  │     Applied with identical weights to:                   │
  │       • focal agent      → focal_embedding   (batch, 8)  │
  │       • each other agent → others_embeddings (batch,N,8) │
  └──────────────────────────────────────────────────────────┘
                          │
  ┌──────────────────────────────────────────────────────────┐
  │  2. Scaled Dot-Product Attention                         │
  │     query = W_q · focal_embedding          (batch, 8)    │
  │     scores = others_embeddings · query^T / √8            │
  │     weights = softmax(scores)              (batch, N)    │
  │     context = weights · others_embeddings  (batch, 8)    │
  │                                                          │
  │     If N=0: context = zeros(8)                           │
  └──────────────────────────────────────────────────────────┘
                          │
  ┌──────────────────────────────────────────────────────────┐
  │  3. Decision Network (ρ)                                 │
  │     input = [fruit_level, focal_level, focal_dist,       │
  │              context]                      (batch, 11)   │
  │     nn.Linear(11, 16) → ReLU → nn.Linear(16, 1) → σ      │
  │     output ∈ [0, 1]                        (batch, 1)    │
  └──────────────────────────────────────────────────────────┘
```

**Key properties:**
- **Shared encoder**: The same 2→8 network encodes both the focal agent and all other agents, forcing the network to learn a single representation rather than memorizing slot-specific patterns.
- **Attention (query)**: The query projection `W_q` is a learned linear layer that transforms the focal agent's embedding into a relevance-matching vector. It learns to answer: *"given who the focal agent is (level, distance to fruit), what kind of other agent would influence this prediction?"* The hypothesis: The dot product between the query and each other agent's embedding should produce a relevance score — e.g. when the focal agent is weak and far away, the network might learn that strong, nearby agents are highly relevant because they would compete for the fruit.
- **Permutation invariance**: No positional encoding, so the output is identical regardless of agent ordering.
- **N-th order beliefs**: All agents visible in one pass enables reasoning like *"Agent A will target this fruit because Agent B is far away and won't compete."*

#### Why supervised and not RL?

Walking is automated via A* pathfinding — the only decision is *which fruit to target*. The NN estimates target probabilities directly. Ground truth labels are available for free whenever a fruit is loaded.

### 3. Target Selection (Expected Reward)

Using the NN predictions from step 2, the agent selects the fruit that maximizes expected reward.

#### Dynamic threshold

Before evaluating fruits, a threshold filters out low-probability helpers:

```
τ = max(Q1(predictions), 1 / |feasible_fruits|)
```

- `Q1` = 25th percentile of this agent's current-round NN predictions
- `1/|feasible_fruits|` = "better than random" baseline

Only agents with `P > τ` are considered as potential helpers.

#### Expected reward (solo)

When `agent_level >= fruit_level`, the agent can load the fruit alone:

```
E[R] = agent_level × fruit_level
```

#### Expected reward (cooperative)

When cooperation is needed, enumerate all subsets S of filtered candidate helpers:

```
E[R] = Σ_S  P(S) × R(S)
```

where:
- `P(S) = Π_{i∈S} P_i × Π_{j∉S} (1 − P_j)` — probability that exactly subset S shows up
- `R(S) = agent_level × fruit_level` if `agent_level + Σ_{i∈S} level_i ≥ fruit_level`, else `0`

No distance penalty is applied — the NN already encodes distance in its inputs.

### 4. Walking and Loading

Once a target fruit is selected, the agent uses A* pathfinding to walk to the closest free loading slot adjacent to the fruit. When it arrives, it issues a LOAD action. The environment rewards all agents adjacent to a fruit when the sum of their levels meets the fruit's level:

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
