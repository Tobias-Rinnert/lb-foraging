# Plan: Survival Simulation with AgentPredictor Architecture

## Context

An older commit implemented survival mechanics (hunger, death, food growth, neuroevolution, terrain) but was based on a flat `build_nn()` architecture that no longer exists. The NN is now an attention-based `AgentPredictor` with `embedding_dim` and `decision_hidden` as its hyperparameters. This plan reimplements the survival simulation adapted to the new architecture.

The frontend renderer already has terrain rendering, hunger bars, and dead-agent ghosts from the old commit. The types, serializer, settings, and GameBoard import are not yet wired up.

---

## Phase 1 — Foundation (independent, parallelizable)

### 1a. New file: `tr_lbf_addon/map_generator.py`

```python
def generate_ca_map(field_size, grass_ratio=0.70, smooth_iterations=5, seed=None) -> np.ndarray
```
- Returns `(field_size, field_size)` int8 array: 0=stone, 1=grass
- Random init with `grass_ratio`, then CA smoothing: cell becomes grass if >=50% of 3x3 neighbourhood is grass
- Tests in `test_map_generator.py` (6 tests: shape, dtype, values, ratio, seed determinism, different seeds differ)

### 1b. Additions to `tr_lbf_addon/neuroevolution.py`

Keep existing `AgentPredictor` unchanged. Add below it:

**`mutate_predictor_dims(embedding_dim, decision_hidden, prob=0.3, min_dim=4, max_dim=64) -> tuple[int, int]`**
- With probability `prob`, shift each dim by random choice from `{-2,-1,+1,+2}`, clamp to `[min_dim, max_dim]`
- Returns `(new_embedding_dim, new_decision_hidden)`

**`transfer_predictor_weights(src: AgentPredictor, dst: AgentPredictor) -> None`**
- Copies overlapping weight slices for each submodule under `torch.no_grad()`:
  - `agent_encoder[0]`: Linear(2, emb). Copy `[:min_emb, :2]` weights, `[:min_emb]` bias
  - `attention_query`: Linear(emb, emb). Copy `[:min_emb, :min_emb]` weights, `[:min_emb]` bias
  - `decision_net[0]`: Linear(3+emb, dh). Copy first 3 columns directly, then context columns `[3:3+min_emb]`, rows `[:min_dh]`
  - `decision_net[2]`: Linear(dh, 1). Copy `[0, :min_dh]` weights, `[0]` bias

**`AgentGenome` dataclass**
- Fields: `agent_id`, `embedding_dim=8`, `decision_hidden=16`, `fitness=0.0`, `nn_model=None`, `optimizer=None`

**`reproduce(parent_genomes, food_eaten_counts, foods_per_child) -> list[AgentGenome]`**
- Each parent gets `floor(food_eaten / foods_per_child)` children
- Each child: mutate dims -> new AgentPredictor -> transfer_predictor_weights -> fresh Adam
- Children get sequential IDs starting at 0
- No `input_size` param needed (AgentPredictor input dims are architecture-invariant)

Tests in `test_neuroevolution.py`: mutate bounds, transfer same/grow/shrink dims, decision_net fixed columns, genome defaults, reproduce basic/no-food/sequential-ids/valid-models

### 1c. Changes to `tr_lbf_addon/lbf_elements.py`

- Add `self.is_alive: bool = True` to `Agent.__init__`
- Add `self.embedding_dim: int = 8` and `self.decision_hidden: int = 16` to `Agent.__init__`
- Dead guard at top of `choose_next_action`: `if not self.is_alive: return np.int64(0)`
- Update `init_neural_network` to pass `embedding_dim` and `decision_hidden` to `AgentPredictor()`

Tests: dead agent returns 0, custom dims create correct model, default is_alive is True

---

## Phase 2 — Integration (depends on Phase 1)

### 2a. Changes to `tr_lbf_addon/lbf_gym.py`

- `update_observation(observation, food_growth=None, dead_agents=None, ca_map=None)` — pass through to get_fruit_infos, update_agents
- `get_fruit_infos(food_growth=None)` — skip fruit if `food_growth.get(pos, 0.0) < 1.0`
- `initialize_agents(agent_infos, ca_map=None)` — pass ca_map to create_path_finding_grid
- `update_agents(new_player_infos, dead_agents=None, ca_map=None)` — dead agents: only update position, skip cognition/learning
- `create_path_finding_grid(agent, ca_map=None)` — add `path_finding_grid[ca_map == 0] = 0` for stone cells
- `agents_choose_actions(fallback_to_closest=True, dead_agents=None)` — dead agents get action 0

Tests: food growth filter hides/shows, dead agent skips cognition, stone cells are obstacles, dead agents skip actions

### 2b. Changes to `tr_lbf_addon/game_runner.py`

**`default_params()`** — add 7 keys:
```
hunger_rate: 0.001, crowding_radius: 3, crowding_penalty: 0.0002,
food_growth_rate: 0.005, foods_per_child: 3, grass_ratio: 0.70, ca_smooth_iterations: 5
```

**`__init__`** — add state:
```python
self.params["initial_number_players"] = params.get("number_players", 5)
self.ca_map = None           # np.ndarray | None
self.food_growth = {}        # dict[tuple, float]
self.agent_hunger = {}       # dict[int, float]
self.agent_food_eaten = {}   # dict[int, int]
self.dead_agents = set()     # set[int]
self._evolved_genomes = []   # list[AgentGenome]
self._next_player_count = params.get("number_players", 5)
```

**`reset()`** — rewrite:
1. Set `params["number_players"] = _next_player_count`, call `_rebuild_env_if_player_count_changed()`
2. `env.reset()`, generate CA map if None
3. Create LBF_GYM, set max levels/distance on agents
4. Inject `_evolved_genomes` into matching agents (set embedding_dim, decision_hidden, nn_model, optimizer)
5. Reset hunger/food_eaten/dead_agents
6. Pre-populate `food_growth` with 1.0 for all food at episode start (bug fix: without this agents are blind for 1/food_growth_rate steps)
7. Standard reset (step_count, rewards, episode_over)

**`step()`** — rewrite:
1. `lbf_gym.update_observation(obs, food_growth, dead_agents, ca_map)`
2. `agents_choose_actions(fallback_to_closest, dead_agents)`
3. `env.step()`
4. Food growth: track new/deleted food positions, grow by `food_growth_rate` per step on grass cells
5. Eating: `reward > 0` -> reset hunger, increment food_eaten
6. Hunger: `hunger_rate + nearby_count * crowding_penalty` per step; `>= 1.0` -> dead
7. Accumulate rewards, record losses
8. Terminate if `terminated or truncated or alive_count == 0`

**`evolve()`** — new method:
1. Build parent genomes from alive agents with NNs
2. Call `reproduce(parents, food_eaten, foods_per_child)`
3. Fallback if no children: each survivor gets 1 mutated child
4. Total extinction: clear genomes, reset to initial_number_players, regenerate terrain
5. Set `_evolved_genomes`, `_next_player_count`

**`rebuild()`** — add: clear genomes, ca_map=None, reset _next_player_count, set initial_number_players

**`_rebuild_env_if_player_count_changed()`** — new: compare `_last_registered_players` vs `_next_player_count`, close/rebuild env if different

---

## Phase 3 — Frontend wiring (depends on Phase 2)

### 3a. `web/backend/serializer.py`

Add to agent dicts:
```python
"is_alive": bool(getattr(agent, "is_alive", True)),
"hunger": float(runner.agent_hunger.get(agent.id, 0.0)),
"food_eaten": int(runner.agent_food_eaten.get(agent.id, 0)),
"nn_architecture": {"embedding_dim": ..., "decision_hidden": ...},
```

Add to frame dict:
```python
"ca_map": runner.ca_map.tolist() if runner.ca_map is not None else None,
"food_growth": {f"{k[0]},{k[1]}": round(float(v), 4) for k, v in runner.food_growth.items()},
"dead_agents": list(runner.dead_agents),
"population_size": len(runner.lbf_gym.agents) if runner.lbf_gym else 0,
"next_population_size": runner._next_player_count,
```

Filter private keys from params: `if not k.startswith("_")`

### 3b. `web/backend/server.py`

- Add `runner.evolve()` before `runner.reset()` in game_loop (line 59) and step handler (line 96)
- Add int casting for `crowding_radius`, `ca_smooth_iterations`, `foods_per_child`
- Add float casting for `hunger_rate`, `crowding_penalty`, `food_growth_rate`, `grass_ratio`

### 3c. `web/frontend/src/types/game.ts`

AgentState: add `is_alive?`, `hunger?`, `food_eaten?`, `nn_architecture?`
GameParams: add 7 survival params
GameFrame: add `ca_map?`, `food_growth?`, `dead_agents?`, `population_size?`, `next_population_size?`

### 3d. `web/frontend/src/components/SettingsPanel.tsx`

- Add `"float"` to FieldDef type union, add `step?` property
- Add "Survival & Evolution" section with all 7 params
- Add float input rendering in JSX

### 3e. `web/frontend/src/components/GameBoard.tsx`

- Import `drawTerrainMap` from renderer
- Call `drawTerrainMap(ctx, frame.ca_map ?? null, frame.food_growth ?? {}, frame.field_size, cell, panX, panY)` after `drawGrid`, before `drawFreeSlots`

---

## Key files

| File | Action |
|------|--------|
| `tr_lbf_addon/map_generator.py` | NEW |
| `tr_lbf_addon/neuroevolution.py` | Add mutate, transfer, genome, reproduce |
| `tr_lbf_addon/lbf_elements.py` | Add is_alive, embedding_dim, decision_hidden, dead guard |
| `tr_lbf_addon/lbf_gym.py` | Add food_growth filter, dead agents, ca_map pathfinding |
| `tr_lbf_addon/game_runner.py` | Major: survival state, hunger, food growth, evolve |
| `web/backend/serializer.py` | Add survival fields |
| `web/backend/server.py` | Add evolve() calls, param casting |
| `web/frontend/src/types/game.ts` | Add survival types |
| `web/frontend/src/components/SettingsPanel.tsx` | Add float type, survival section |
| `web/frontend/src/components/GameBoard.tsx` | Import and call drawTerrainMap |
| `tr_lbf_addon/tests/test_map_generator.py` | NEW (6 tests) |
| `tr_lbf_addon/tests/test_neuroevolution.py` | Add mutation/transfer/reproduce tests |
| `tr_lbf_addon/tests/test_lbf_elements.py` | Add is_alive, custom dims tests |
| `tr_lbf_addon/tests/test_lbf_gym.py` | Add food growth, dead agent, stone cell tests |

## Verification

```bash
python -m pytest tr_lbf_addon/tests/ -v \
  --ignore=tr_lbf_addon/tests/test_app_board_canvas.py \
  --ignore=tr_lbf_addon/tests/test_app_game_runner.py
```

Then start the web app and verify:
- Terrain renders (stone=grey, grass=green tint)
- Food growth overlay shows fading green
- Agents die when hunger reaches 1.0 (ghost rendering)
- Hunger bars visible and animate green->red
- Population evolves between episodes (count changes)
- Survival params editable in settings panel
