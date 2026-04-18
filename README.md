# TR LBF Foraging

RL agents for the Level-Based Foraging environment, exploring n-order belief estimation for cooperative multi-agent decision making.

## Overview

This project builds on the [Level-Based Foraging (LBF)](https://github.com/semitable/lb-foraging) environment. The idea is to automate walking and loading with A* pathfinding so that the only decision to make is which fruit to target. This decision depends on first-order beliefs about the chosen fruit of the other agents. A neural network estimates these first-order beliefs.

Predictions are made once per agent and reused until that agent deviates from its predicted A* path. This conditional re-prediction strategy reduces NN calls significantly. Target selection uses combinatorial expected reward over all subsets of likely helpers.

Key goals:
- Use the simplest and most efficient models — automate walking so the neural network can focus on estimating the actual stochastic process
- make sim so that different types of agents develop with different strategies: selfish vs cooperating, and test which parameter combinations lead to which agents

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
  map_generator.py      Cellular automata terrain generator (stone/grass maps)
  neuroevolution.py     AgentPredictor architecture, mutation, weight transfer, reproduction
  tests/                Unit tests

web/                    Browser GUI
  backend/              Python WebSocket server (server.py)
  frontend/             React + Vite frontend

lbforaging/             Vendored upstream LBF environment (modified fork)
start.bat               One-click launcher for the web app (Windows)
```

## Agent Behavior Fixes

- **Dead-agent filter** (`process_agent_infos`, `choose_fruit`): dead agents drop out of `known_agents` at the source, so coop-level sums, NN predictions, and expected-reward helper subsets only see live peers. Stale predictions keyed on dead agents are purged in `choose_fruit`.
- **Unreachable-fruit filter** (`choose_fruit`): feasibility filter now also requires `fruit.free_slots` to be non-empty, eliminating the flip-flop where an agent re-picks a blocked fruit every step.
- **Empty `known_fruits` guard** (`choose_fruit`): short-circuits to `self.target = None` before the `np.all(..., axis=1)` call that crashed with `AxisError` on a 1-D empty array.
- **Walking-agents-are-not-obstacles** (`choose_next_action`): only loading agents block available slots; walking agents are allowed to occupy the route and move away by arrival time.
- **`record_ground_truth` O(n²) → O(n) lookup**: precomputed `agent_positions_by_id` dict replaces a per-prediction linear scan; adjacency check vectorized via numpy broadcasting.
- **Dead `ca_map` check in `food_growth` removed**: fruits can only spawn on grass, so the runtime stone-guard was unreachable; removed with a comment documenting the invariant.

## Changes to LBF

The `lbforaging/` directory is a slightly modified fork of the original environment:
- Added agent IDs for tracking
- Added `full_info_mode` to get all information from the observation conveniently
- Changed collision logic: when two agents want the same cell, one fails randomly while the other succeeds
- Changed agent spawn to always start at level 1 (was random between `min_player_level` and `max_player_level`)
- Added level-up mechanic: every agent that participates in a successful food load gains +1 level (capped at `max_player_level`)
- Added `min_level_1_food` parameter (default 2): after `spawn_food`, forces at least that many fruits to level 1 so level-1 agents always have accessible food
- Added `ca_map` attribute to `ForagingEnv`: when set externally before `reset()`, `_is_empty_location` rejects stone cells (`ca_map == 0`) so fruits and players never spawn on impassable terrain

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
- **Configurable dimensions**: `embedding_dim` and `decision_hidden` are per-agent hyperparameters (defaults: 8 and 16). They can be set from an evolved genome before `init_neural_network()` is called, allowing the neuroevolution system to vary architecture sizes across agents.
- **Survival guard**: Dead agents (`is_alive = False`) skip all cognition, learning, and action selection — `choose_next_action()` immediately returns `0` for dead agents.

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

## Survival Simulation

Between episodes, agents age, starve, and reproduce. The mechanics live in `game_runner.py`.

### Hunger and Death

Each step, every alive agent's hunger increases by `hunger_rate`. When an agent eats (reward > 0), its hunger resets to 0.0. When hunger reaches 1.0, the agent is added to `dead_agents` and its `is_alive` flag is set to `False`. Dead agents skip all cognition, learning, and action selection for the rest of the episode.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hunger_rate` | 0.001 | Hunger increase per step |

### Food Growth

When new fruits appear in the environment, they start hidden (`food_growth = 0.0`) and only become visible to agents once their growth value reaches 1.0. Each step, fruits on grass cells grow by `food_growth_rate`. Fruits on stone cells do not grow.

Fruits present at episode start are pre-populated at 1.0 so agents are not blind at the beginning.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `food_growth_rate` | 0.005 | Growth increment per step (on grass) |

### Terrain (CA Map)

`map_generator.generate_ca_map()` generates a `(field_size, field_size)` binary terrain map at the start of the first episode (reused across subsequent episodes, regenerated on total extinction). Terrain is shared with `lbf_gym.py`: stone cells (`ca_map == 0`) become obstacles in the A* pathfinding grid.

### Neuroevolution

Agents evolve between episodes. The building blocks live in `neuroevolution.py`.

### `mutate_predictor_dims`

Each dimension (`embedding_dim`, `decision_hidden`) is independently mutated with probability `prob` (default 0.3) by adding a random delta from `{-2, -1, +1, +2}`, clamped to `[min_dim, max_dim]` (default 4–64).

### `transfer_predictor_weights`

Copies overlapping weight slices from a parent `AgentPredictor` to a child `AgentPredictor`. When dimensions differ, only the `[:min_dim]` slice is copied; extra neurons in the child retain random initialisation. This preserves learned representations while allowing the architecture to grow or shrink:

| Layer | Copied slice |
|-------|-------------|
| `agent_encoder[0]` Linear(2, emb) | `[:min_emb, :2]` weights, `[:min_emb]` bias |
| `attention_query` Linear(emb, emb) | `[:min_emb, :min_emb]` weights, `[:min_emb]` bias |
| `decision_net[0]` Linear(3+emb, dh) | first 3 columns (fixed inputs) + `[3:3+min_emb]` (context), rows `[:min_dh]` |
| `decision_net[2]` Linear(dh, 1) | `[0, :min_dh]` weights, `[0]` bias |

### `AgentGenome`

A dataclass bundling everything needed to reconstruct an agent's NN across episodes:

```
agent_id        int     — unique identifier
embedding_dim   int     — per-agent embedding size (default 8)
decision_hidden int     — decision network hidden size (default 16)
fitness         float   — accumulated fitness (e.g. food eaten)
nn_model        AgentPredictor | None
optimizer       Adam | None
```

### `reproduce`

Creates the next generation from a list of parent genomes and a food-eaten count:

```
n_children = floor(food_eaten[parent_id] / foods_per_child)
```

For each child: mutate dims → new `AgentPredictor` → `transfer_predictor_weights` → fresh Adam optimizer. Children get sequential IDs starting at 0.

### `evolve()` — inter-episode lifecycle

Called by the application between episodes:

1. Build parent genomes from alive agents (with NNs) and their food-eaten counts.
2. Call `reproduce()` to create the next generation.
3. **Fallback**: if no parent earned enough food for a child, each survivor produces one mutated child (so the population never collapses due to a hard episode).
4. **Total extinction**: if no survivors, clear genomes, reset player count to `initial_number_players`, and regenerate terrain.
5. Save the best parent genome to `saved_genome.pt`.
6. Set `_evolved_genomes` (injected at next `reset()`) and `_next_player_count`.

### Genome Persistence

The best parent genome (highest food count) is saved to `saved_genome.pt` in the repo root after every episode via `_save_best_genome()`. On the next app start, `_load_saved_genome()` reads it back. On the first `reset()` call, a full initial population is bootstrapped from the saved genome via `reproduce()`, then `_saved_genome` is cleared.

This means trained weights survive restarts and settings changes. The save file is NOT cleared by `rebuild()`.

```
App start → _load_saved_genome() → _saved_genome set
reset()   → reproduce([saved_genome], synthetic_food) → _evolved_genomes populated
episode   → agents learn, eat, starve, die
evolve()  → reproduce(alive_parents) → _save_best_genome() → next generation ready
reset()   → inject _evolved_genomes into new agents
...
```

## Web App — Survival Features

The browser GUI renders the survival simulation in real time.

### Terrain

Both cell types use a diagonal gradient (top-left light source) plus bevel edges for a 3D raised-tile look. Stone cells shade from light grey (#A8A8A8) to dark grey (#505050); grass cells shade from light green (#6DBF71) to dark green (#235E27). Bevel edges are skipped below 6 px cell size. Growing fruits (food_growth < 1.0) show a light-green brightening overlay on their grass cell that strengthens as they approach ripeness.

### Agent Status

Each agent displays a hunger bar below its sprite. The bar fills from green (0 hunger) through yellow to red (hunger → 1.0). Dead agents are rendered as ghosts at 30% opacity.

### Population HUD

The frame payload carries `population_size` (current alive agents) and `next_population_size` (children ready for next episode), exposed for overlay display.

### Agent Levelling

Agents always start at **level 1** on episode reset. Each time an agent successfully loads a fruit (alone or cooperatively), its level increases by 1, up to `max_player_level`. This makes early-episode food availability critical — see `min_level_1_food` below.

### Survival & Evolution Settings

The settings panel has a new **Survival & Evolution** section with all survival parameters. Float inputs support fine-grained editing with configurable step sizes. Changes take effect on the next **Apply & Restart**.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_level_1_food` | 2 | Minimum number of level-1 fruits guaranteed at episode start so level-1 agents always have something to eat |

### End-to-End Flow

```
App start
  └─ _load_saved_genome() → inject saved weights into first generation

Episode loop (Play button):
  step() → update hunger/food_growth → agents act
  episode_over? → evolve() → _save_best_genome() → reset()

Between episodes:
  reproduce(alive_parents) → new generation
  inject genomes into agents via reset()
  ca_map reused (regenerated on total extinction)
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

## License

This project is source-available with an **academic publication restriction**. You may use, modify, and distribute the code, but only Tobias Rinnert may publish academic papers or scholarly works based on it. See [LICENSE](LICENSE) for details.

## Attribution

The LBF environment is by Filippos Christianos et al. See `lbforaging/LICENSE` for the original license.

## TODOs

Detailed implementation plans live in [`plans/`](plans/):

- **Stuck-agent & robustness edge cases** — [`plans/make-plan-stuck-edge-cases-to-a-cheeky-koala.md`](plans/make-plan-stuck-edge-cases-to-a-cheeky-koala.md). Mechanical fix list for six issues in agent cognition (dead-agent filter, unreachable-fruit filter, empty `known_fruits` guard, walking-agents-are-not-obstacles, `record_ground_truth` performance, dead `ca_map` check).
- **Interactive NN architecture viewer** — [`plans/plan-nn-architecture-viz.md`](plans/plan-nn-architecture-viz.md). Zoomable, data-driven visualization of each agent's `AgentPredictor` in the web UI. Opens → auto-pauses the game → shows architecture graph, weight heatmaps, and frozen forward-pass activations. Learning metrics stay in `MetricsPanel`.
- **Learned re-prediction gate + time-series NN input** — [`plans/plan-learned-reprediction-gate.md`](plans/plan-learned-reprediction-gate.md). Replaces the hardcoded `is_agent_on_predicted_path` + `_stationary_steps` heuristic with a learned gate, and feeds a time-series of world states so the NN can represent signals like "agent B has been standing still for many steps". Addresses the deferred "adjacent-and-LOADing-forever with absent helper" edge case.

Open ideas without plans yet:
- Use the `AgentPredictor` for the focal agent's own target decision too — reuse the other-agents prediction probabilities as a learned embedding, skipping the input-layer encoding. Unifies self-decision and other-prediction under one model.
- Long-term: a single architecture that predicts any action. The predictor outputs a distribution over actions; re-plan / load / move all share one inference pass.