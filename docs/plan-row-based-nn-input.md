# Plan: Row-Based NN Input (Fixed-Capacity Per-Agent Grid)

## Context

The NN predicts which fruit another agent will target. Its input is currently a flat vector of size
`n_agents * 2 + 1` — it grows with population, forcing NN rebuilds every time agent count changes.

**Solution:** Structure the input as a 2D grid of rows × features. Each agent gets its own row of
neurons. The grid has a fixed capacity K rows. Inactive rows are zeroed. When population grows
beyond K, new rows are appended to the first Linear layer with weights **copied from a random
existing row** (warm start). Architecture only grows — never shrinks.

---

## Input structure (`1 + K×2` features, always fixed)

```
Global (1 neuron):  [ fruit_level ]

Row 0 (focal):      [ level  |  distance ]   ← always the agent being predicted
Row 1:              [ level  |  distance ]   ← other agents sorted by agent_id
Row 2:              [ level  |  distance ]
...
Row K-2:            [ level  |  distance ]
Row K-1:            [ 0      |  0        ]   ← inactive (zeroed when pop < K)
```

Total input = `1 + K * 2` (same formula as today, but K is now a fixed capacity that only grows).

K starts at `initial_number_players`. When population grows to N > K: add N-K rows by copying
weights of a random existing row into the new columns of the first Linear layer.

---

## New function: `add_agent_rows(model, n_new) -> nn.Sequential` in `neuroevolution.py`

Extends the first Linear layer by `n_new * 2` input neurons (one new row = 2 neurons).
New neurons get weights copied from a random existing agent row (not the global neuron at col 0).

```python
def add_agent_rows(model: nn.Sequential, n_new: int) -> nn.Sequential:
    import copy, random, torch
    model = copy.deepcopy(model)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    first = linears[0]                      # shape: [hidden, 1 + K_old*2]
    K_old = (first.in_features - 1) // 2   # existing row count
    with torch.no_grad():
        extra_cols = []
        for _ in range(n_new):
            donor = random.randint(0, K_old - 1)  # pick a random existing row
            col = 1 + donor * 2                    # column offset for that row
            extra_cols.append(first.weight[:, col:col + 2].clone())
        new_weight = torch.cat([first.weight] + extra_cols, dim=1)
        new_linear = nn.Linear(first.in_features + 2 * n_new, first.out_features)
        new_linear.weight = nn.Parameter(new_weight)
        new_linear.bias   = nn.Parameter(first.bias.clone())
    new_layers = []
    replaced = False
    for layer in model:
        if isinstance(layer, nn.Linear) and not replaced:
            new_layers.append(new_linear)
            replaced = True
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)
```

---

## New helper: `_build_row_input(focal_pos, focal_level, others, fruit, known_fruits, grid, n_rows)`

Module-level function in `lbf_elements.py`. Builds the `(1 + n_rows*2,)` input vector.

```python
def _build_row_input(focal_pos, focal_level, others, fruit, known_fruits, grid, n_rows):
    from pathfinding.core.grid import Grid
    from pathfinding.finder.a_star import AStarFinder
    from pathfinding.core.diagonal_movement import DiagonalMovement

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    field_diag = float(grid.shape[0]) * 1.415

    def _dist(p, free_slots):
        if not free_slots:
            return field_diag * 2
        slot = min(free_slots, key=lambda s: np.linalg.norm(s - np.array(p)))
        g = Grid(matrix=grid.tolist())
        path, _ = finder.find_path(g.node(int(p[1]), int(p[0])),
                                   g.node(int(slot[1]), int(slot[0])), g)
        return float(len(path)) if path else field_diag * 2

    # fruit level (global, normalized across all known fruits)
    fls = [f.level for f in known_fruits]
    fruit_norm = (fruit.level - min(fls)) / (max(fls) - min(fls)) if max(fls) != min(fls) else 0.0

    # gather raw values
    focal_dist_raw   = _dist(focal_pos, fruit.free_slots)
    others_dists_raw = [_dist(a["position"], fruit.free_slots) for a in others]

    # normalize levels across all agents (focal + others)
    all_levels = [focal_level] + [a["level"] for a in others]
    lmin, lmax = min(all_levels), max(all_levels)
    def nlvl(l): return (l - lmin) / (lmax - lmin) if lmax != lmin else 0.0

    # normalize distances by field diagonal
    all_dists = [focal_dist_raw] + others_dists_raw
    dmin, dmax = min(all_dists), max(all_dists)
    def ndist(d): return (d - dmin) / (dmax - dmin) if dmax != dmin else 0.0

    vec = np.zeros(1 + n_rows * 2, dtype=np.float32)
    vec[0] = fruit_norm
    # row 0: focal agent
    vec[1] = nlvl(focal_level)
    vec[2] = ndist(focal_dist_raw)
    # rows 1..n_active-1: other agents sorted by agent_id
    others_sorted = sorted(zip(others, others_dists_raw), key=lambda x: x[0].get("id", 0))
    for i, (a, d) in enumerate(others_sorted[:n_rows - 1]):
        vec[3 + i * 2] = nlvl(a["level"])
        vec[3 + i * 2 + 1] = ndist(d)
    # rows n_active..n_rows-1 stay 0
    return vec
```

---

## Files to modify

### 1. `tr_lbf_addon/neuroevolution.py`

- Add `add_agent_rows(model, n_new)` function
- Add `n_rows: int = 5` field to `AgentGenome`
- Update `reproduce(parent_genomes, food_eaten_counts, foods_per_child, n_rows_next)`:
  - For each child: if `n_rows_next > parent.n_rows` → call `add_agent_rows(parent.nn_model, n_rows_next - parent.n_rows)` before `build_nn` + `transfer_weights`
  - Pass `input_size = 1 + n_rows_next * 2` to `build_nn`
  - Set `child.n_rows = n_rows_next`

### 2. `tr_lbf_addon/lbf_elements.py`

- Add `_build_row_input(...)` at module level
- Add `self._n_rows: int = 5` to `Agent.__init__`
- **`get_training_data_per_fruit(fruit)`**: replace DataFrame body with:
  ```python
  return _build_row_input(self.position, self.level, self.known_agents or [],
                          fruit, self.known_fruits, self.path_finding_grid, self._n_rows)
  ```
  Return type: `np.ndarray(1 + K*2,)`. The `learn()` method is unchanged.
- **`predict_agent_target(agent_id, feasible_fruits)`**: build input from focal agent's view:
  ```python
  focal_info = next((a for a in (self.known_agents or []) if a["id"] == agent_id), None)
  if focal_info is None: return feasible_fruits[0] if feasible_fruits else None
  others = [a for a in (self.known_agents or []) if a["id"] != agent_id]
  others += [{"id": self.id, "position": self.position.tolist(), "level": self.level}]
  for fruit in feasible_fruits:
      nn_input = _build_row_input(np.array(focal_info["position"]), focal_info["level"],
                                  others, fruit, self.known_fruits,
                                  self.path_finding_grid, self._n_rows).reshape(1, -1)
      # rest unchanged (predict, append to self.predictions, track best)
  ```
- **`init_neural_network()`**: use `1 + self._n_rows * 2` as input size:
  ```python
  from neuroevolution import build_nn
  model = build_nn(1 + self._n_rows * 2, self.nn_architecture)
  self.neural_network = model
  self.optimizer = torch.optim.Adam(model.parameters())
  ```
- Remove `import pandas as pd` if no longer used elsewhere

### 3. `tr_lbf_addon/game_runner.py`

- **`__init__`**: add `self._n_rows: int = params.get("number_players", 5)`
- **`reset()` injection block**: inject `_n_rows` into agents; simplify (no size-check needed):
  ```python
  for i, genome in enumerate(self._evolved_genomes):
      if i < len(self.lbf_gym.agents):
          a = self.lbf_gym.agents[i]
          a.neural_network  = genome.nn_model
          a.optimizer       = genome.optimizer
          a.nn_architecture = genome.hidden_layers
          a._n_rows         = genome.n_rows
  ```
- **`evolve()`**: compute `n_rows_next` before reproducing:
  ```python
  n_children_total = sum(
      self.agent_food_eaten.get(g.agent_id, 0) // max(1, self.params["foods_per_child"])
      for g in parent_genomes
  ) or len(alive)  # fallback if no one reproduced
  n_rows_next = max(self._n_rows, n_children_total)  # only grows
  self._n_rows = n_rows_next
  children = reproduce(parent_genomes, self.agent_food_eaten,
                       self.params["foods_per_child"], n_rows_next)
  ```
  In the fallback (no-food path), pass `n_rows_next = self._n_rows` to keep rows stable.
- **`rebuild()`**: reset `_n_rows` to initial players on full restart

### 4. `tr_lbf_addon/tests/test_neuroevolution.py`

- Add tests for `add_agent_rows`:
  - `test_add_agent_rows_increases_input` — output NN has correct new input size
  - `test_add_agent_rows_copies_donor_weights` — new columns match a donor row
  - `test_add_agent_rows_preserves_other_weights` — existing columns unchanged
- Update `reproduce(...)` calls to pass `n_rows_next=5`
- Update `AgentGenome(...)` calls to include `n_rows=5`

---

## Implementation order

1. `neuroevolution.py` — add `add_agent_rows`, update `AgentGenome`, update `reproduce`
2. `lbf_elements.py` — add `_build_row_input`, update `Agent.__init__`, update 3 methods
3. `game_runner.py` — add `_n_rows`, update `reset`, `evolve`, `rebuild`
4. `test_neuroevolution.py` — add `add_agent_rows` tests, update existing calls
5. Run all tests + smoke test

---

## Verification

```bash
python -m pytest tr_lbf_addon/tests/ -v

python -W ignore -c "
import sys, torch.nn as nn
sys.path.insert(0, 'tr_lbf_addon')
from game_runner import GameRunner, default_params
params = default_params()
params.update(max_episode_steps=200, hunger_rate=0.005,
              foods_per_child=1, food_growth_rate=0.5, field_size=12)
runner = GameRunner(params)
for ep in range(5):
    runner.reset()
    while not runner.episode_over: runner.step()
    for a in runner.lbf_gym.agents:
        if a.neural_network:
            lin = [m for m in a.neural_network.modules() if isinstance(m, nn.Linear)][0]
            expected = 1 + runner._n_rows * 2
            assert lin.in_features == expected, f'Expected {expected}, got {lin.in_features}'
    print(f'ep {ep+1}: pop={len(runner.lbf_gym.agents)}, n_rows={runner._n_rows}, input={1+runner._n_rows*2} OK')
    runner.evolve()
print('PASSED')
"
```
