# Plan — Fix Remaining Stuck-Agent & Robustness Edge Cases

Covers issues 1–7 from the game-loop audit (Issue 5 is deferred to
`plan-learned-reprediction-gate.md`). Each section is self-contained:
**problem → root cause → fix → files → tests → risks**.

Follow the project coding workflow throughout: descriptive naming, for-loops over
while-loops, vectorized numpy where possible, docstrings on every new/changed
function, add tests only where coverage is missing, run the full suite at the end,
update the README.

Suggested execution order: **3 → 2 → 1 → 4 → 5 → 6 → 7**. Rationale: issue 3
(dead-agent filtering) eliminates several downstream symptoms, so fixing it first
simplifies the reproduction cases for issues 2 and 4.

---

## Issue 1 — `choose_fruit` crashes when `known_fruits` is empty

### Problem
`AxisError` in `lbf_elements.py:355-359` when the last fruit gets loaded while
an agent still holds a stale `self.target`.

### Root Cause
```python
fruit_positions = np.array([fruit.position for fruit in self.known_fruits])
current_target_still_in_game = (
    self.target is not None
    and np.any(np.all(self.target.position == fruit_positions, axis=1))
)
```
When `known_fruits == []`, `fruit_positions.shape == (0,)` (1-D). Calling
`np.all(..., axis=1)` on a 1-D array raises `numpy.exceptions.AxisError`.

### Fix
Short-circuit at the top of `choose_fruit`:

```python
if not self.known_fruits:
    self.target = None
    return
```

Place this **before** the `fruit_positions` construction. This also obviates the
need for further empty-list handling later in the function (prediction
invalidation, selection) because those paths are naturally no-ops on empty input,
but the early return keeps the code shape clean.

### Files
- `tr_lbf_addon/lbf_elements.py` — `Agent.choose_fruit` (~2 lines added)

### Tests (add in `test_lbf_elements.py`, `TestChooseFruit`)
1. `test_empty_known_fruits_clears_target_without_crash`
   - Given `agent.known_fruits = []` and `agent.target = Fruit(...)`
   - Call `choose_fruit()`
   - Assert `agent.target is None` and no exception raised
2. `test_empty_known_fruits_with_force_reselect`
   - Same setup with `force_reselect=True`; verify identical behavior

### Risks
- Minimal. The early-return path already exists conceptually (selection returns
  None on empty feasible list). We're guarding one arithmetic line.

---

## Issue 2 — Unreachable fruits stay in `feasible_fruits` → flip-flop stuck

### Problem
Agent commits to a fruit whose `free_slots == []`, `choose_next_action` clears
the target, next step re-selects the same fruit, forever idle.

### Root Cause
`choose_fruit` feasibility filter only checks level:
```python
feasible_fruits = [
    fruit for fruit in self.known_fruits
    if fruit.level <= max_achievable
]
```
A fruit with zero free slots (all slots occupied by other agents or blocked) is
still feasible by level, so `select_fruit_by_expected_reward` picks it with
positive EV (`self.level * fruit.level`).

### Fix
Add a reachable-slot check:
```python
feasible_fruits = [
    fruit for fruit in self.known_fruits
    if fruit.level <= max_achievable and fruit.free_slots
]
```
One extra conjunction. This mirrors the filter already used in
`_fallback_target` (`[f for f in agent.known_fruits if f.free_slots]`),
bringing both paths into agreement.

Note: `free_slots` is recomputed every tick in `get_fruit_infos`, so a fruit
that becomes reachable again (an agent moves off its slot) re-enters
`feasible_fruits` automatically on the next tick.

### Files
- `tr_lbf_addon/lbf_elements.py` — `Agent.choose_fruit` (1 line changed)

### Tests (add in `test_lbf_elements.py`, `TestChooseFruit`)
1. `test_fruit_without_free_slots_excluded_from_feasible`
   - Build known_fruits with two fruits, one `free_slots=[]`, one with slots
   - Assert `choose_fruit` selects the one with slots
2. `test_all_fruits_without_free_slots_yields_none_target`
   - All `known_fruits` have empty `free_slots`
   - Assert `agent.target is None` after `choose_fruit`
3. `test_previously_blocked_fruit_selectable_after_slots_free_up`
   - Two sequential `choose_fruit` calls; mutate `free_slots` between them;
     verify second call picks the now-reachable fruit

### Risks
- Low. A fruit with no free slots cannot be loaded anyway, so excluding it from
  selection strictly improves behavior.
- Corner case: if ALL known fruits are temporarily blocked (e.g., heavy
  congestion for one tick), `feasible_fruits == []` → `target = None` → agent
  idles this step. Next step slots may reopen. This is correct behavior.

---

## Issue 3 — Dead agents pollute cooperation & prediction logic

### Problem
Dead agents remain in `self.agents` and flow through `process_agent_infos` into
every agent's `known_agents`. They distort three distinct subsystems:
1. **Coop-level sums** — `get_possible_coop_level_sums` inflates
   `max_achievable` using dead agents' levels.
2. **NN predictions** — `predict_agent_target` predicts dead agents' targets;
   `is_agent_on_predicted_path` returns True (they're pinned to their death
   position) so the prediction is never refreshed.
3. **Expected reward** — `select_fruit_by_expected_reward` uses dead agents as
   candidate helpers with non-zero probability.

Result: a living agent walks to a coop fruit, LOADs forever, waiting for a
"helper" that is dead.

### Fix
Centralize the liveness filter at the **source** so all three subsystems see
only live helpers without scattering `is_alive` checks.

**Location:** `Agent.process_agent_infos` in `lbf_elements.py`.

**Change:** filter out agents with `is_alive == False` (and keep excluding self,
as before):
```python
self.known_agents = [
    {
        "id": agent.id,
        "position": agent.position,
        "level": agent.level,
        "position_history": agent.position_history[-self.memory_size:],
        "last_action": agent.last_action,
        "is_loading": agent.is_loading,
    }
    for agent in agents
    if agent.id != self.id and agent.is_alive
]
```

**Why this location works:**
- `get_possible_coop_level_sums` reads from `known_agents` → dead excluded.
- `predict_agent_target` / `select_fruit_by_expected_reward` iterate
  `known_agents` → dead excluded.
- `choose_next_action`'s `other_positions` set iterates `known_agents` → dead
  agents' bodies stop blocking slots. *(See Risks below — we need to decide
  whether dead bodies should still block.)*

**Cleanup of stale predictions** (secondary change in `choose_fruit`):
When an agent dies, its entries in `predicted_targets`, `predicted_paths`,
`prediction_round` become stale. Add a pass alongside the existing
"invalidate predictions for removed fruits" loop:
```python
live_agent_ids = {a["id"] for a in self.known_agents}
for agent_id in list(self.predicted_targets.keys()):
    if agent_id not in live_agent_ids:
        del self.predicted_targets[agent_id]
        del self.predicted_paths[agent_id]
        del self.prediction_round[agent_id]
```

### Decision point — do dead bodies block movement?
- **A (simpler):** dead agents do not block anything. Filter at `known_agents`,
  also skip them in `create_path_finding_grid` loading-agent check (they're
  `is_loading == False` so already skipped — no change needed).



### Files
- `tr_lbf_addon/lbf_elements.py` — `Agent.process_agent_infos` (~1 line),
  `Agent.choose_fruit` (~5 lines)

### Tests (add in `test_lbf_elements.py`)
Create new class `TestDeadAgentFiltering`:
1. `test_dead_agent_excluded_from_known_agents`
   - Build 3 agents, mark agent 2 `is_alive=False`, call `process_agent_infos`
     on agent 0 → `known_agents` contains only agent 1
2. `test_dead_agent_level_not_in_coop_sums`
   - Verify `get_possible_coop_level_sums` drops a dead high-level agent's
     contribution after `process_agent_infos`
3. `test_dead_agent_prediction_invalidated_on_choose_fruit`
   - Pre-populate `predicted_targets[2]`, `predicted_paths[2]`,
     `prediction_round[2]`; mark agent 2 dead; call `choose_fruit`; verify all
     three dicts no longer contain key 2
4. `test_dead_agent_not_candidate_helper_in_expected_reward`
   - Cooperative fruit, one dead high-level would-be helper, one live
     low-level helper; verify EV uses only live helper

### Risks
- **Predicted_paths trimming:** if a dead agent appears in another living
  agent's existing `predicted_paths`, we must drop it. Covered above.
- **`record_ground_truth`** labels predictions based on adjacency of the
  *predicted agent*. If the predicted agent is now dead (filtered out of
  `known_agents`), no new predictions are made, and old predictions in
  `self.predictions` still carry their `agent_id`. Loading a fruit that was
  predicted for a now-dead agent would still label those predictions. This is
  benign — training on the last known state of that agent doesn't hurt.
- **All helpers die mid-episode:** agent reduced to solo-only feasibility.
  Correct behavior.

---

## Issue 4 — All slots blocked by other *walking* agents → flip-flop (1-step)

### Problem
`choose_next_action` clears the target when `available_slots` (free_slots minus
other agents' current positions) is empty. Next step re-picks the same fruit
with no memory of the blockage. Waste of one turn; not permanent since walking
agents move next step.

### Root Cause
Same class as Issue 2 but at a finer granularity — the `other_positions`
filter runs *after* `free_slots` has already been validated.

### Fix
Rather than clearing the target, **attempt to path to ANY slot**, including
slots currently occupied by walking (non-loading) agents. Walking-agent slots
are walkable in `path_finding_grid` (only loading agents are marked
obstacles), so A* can route there. By the time the focal agent arrives, the
walking occupant will have moved.

Change in `choose_next_action`:
```python
# Exclude slots occupied by LOADING agents (they won't move);
# walking agents are allowed — A* plans through them, env collision handles the rest.
loading_positions = {
    tuple(a["position"])
    for a in (self.known_agents or [])
    if a.get("is_loading")
}
available_slots = [s for s in self.target.free_slots if tuple(s) not in loading_positions]
if not available_slots:
    self.target = None
    return np.int64(0)
```

This matches the semantics already used in `create_path_finding_grid`.

### Files
- `tr_lbf_addon/lbf_elements.py` — `Agent.choose_next_action` (~5 lines changed)

### Tests (add/modify in `test_lbf_elements.py`, `TestChooseNextAction`)
1. `test_walking_agent_on_slot_does_not_clear_target`
   - Fruit with slot S; other agent (`is_loading=False`) at S; focal agent
     targets fruit; assert target stays set and action is a movement toward S
2. `test_loading_agent_on_slot_excluded_from_available`
   - Fruit with two slots; a loading agent on the closest; focal agent paths
     to the farther slot
3. `test_all_slots_occupied_by_loading_agents_clears_target`
   - Preserve existing clear-target behavior when truly impassable

### Risks
- **Collision churn:** two walking agents both planning into each other's
  cells. Env collision logic handles this (both fail to move). Stationary
  recovery kicks in after 3 steps. Acceptable.
- **No regression on currently-passing tests:** verify none assume
  walking-agent-clears-target semantics.

---

## Issue 5 — Adjacent-and-LOADing-forever with absent helper

**Status: MOVED OUT OF THIS PLAN.**

The failed-LOAD hardcoded-threshold fix originally proposed here was rejected
because it contradicts the "no hardcoded thresholds" direction the project is
taking. The root cause — the NN has no negative-feedback signal for failed
cooperation, and the re-prediction gate is a hand-coded heuristic — is now
addressed architecturally in **`plan-learned-reprediction-gate.md`**, together
with the README todo for time-series NN input that gives the main NN a way to
represent "agent B has not moved for many steps" in the first place.

Interim mitigation until that plan ships:
- Issue 2 (filter unreachable fruits) eliminates most stuck-LOADing scenarios
  by dropping fruits with empty `free_slots` from feasibility.
- Issue 3 (filter dead agents out of `known_agents`) eliminates the dead-helper
  variant.
- Existing stationary recovery at `_STATIONARY_RESELECT_THRESHOLD = 3` acts as
  a safety net, forcing reselection.
- Evolutionary pressure selects against NNs that produce persistent false-helper
  predictions — stuck agents do not eat, so they do not reproduce.

Nothing further to implement under Issue 5 in this plan.

---

## Issue 6 — `record_ground_truth` is `O(agents × predictions × agents)`

### Problem
Three nested loops for every loaded fruit. Not a stuck bug — pure performance.

### Root Cause
`record_ground_truth:347-361`:
```python
for agent in self.agents:
    for prediction in agent.predictions:
        ...
        predicted_agent = next(
            (a for a in self.agents if a.id == prediction["agent_id"]), None
        )
```
The `next(...)` scan over `self.agents` repeats inside the inner loop.

### Fix
Precompute a position lookup by agent id once per call:
```python
agent_positions_by_id = {a.id: a.position for a in self.agents}
```
Inside the inner loop, replace the `next(...)` scan with:
```python
predicted_position = agent_positions_by_id.get(prediction["agent_id"])
if predicted_position is None:
    continue
was_adjacent = any(
    np.array_equal(predicted_position, slot) for slot in adjacent_slots
)
```
Also vectorize the adjacency check:
```python
adjacent_slots_arr = np.array(adjacent_slots)  # shape (4, 2)
...
was_adjacent = bool(np.any(np.all(adjacent_slots_arr == predicted_position, axis=1)))
```

Move `adjacent_slots_arr` construction outside the inner loops (it's per-fruit,
not per-prediction).

### Files
- `tr_lbf_addon/lbf_gym.py` — `LBF_GYM.record_ground_truth` (~10 lines)

### Tests
No new tests required — existing tests in `TestRecordGroundTruth`
(if present in `test_lbf_gym.py`) must still pass. If no direct coverage
exists, add:
1. `test_ground_truth_labels_match_before_and_after_refactor`
   - Build a fixture with predictions + loaded fruit, snapshot expected
     labels, verify refactor preserves them

### Risks
- Pure refactor, observable behavior unchanged. Verify test suite green.

---

## Issue 7 — Defensive `ca_map[pos] == 1` check in `food_growth`

### Problem
`game_runner.py:232-234`:
```python
for pos in new_fruit_positions & old_positions:
    if self.ca_map is None or self.ca_map[pos] == 1:
        self.food_growth[pos] = min(1.0, self.food_growth[pos] + food_growth_rate)
```
Fruits never spawn on stone cells (enforced by `_is_empty_location` in
`environment.py`). Check is structurally unreachable.

### Fix (two options)
**Option A (recommended):** remove the check:
```python
for pos in new_fruit_positions & old_positions:
    self.food_growth[pos] = min(1.0, self.food_growth[pos] + food_growth_rate)
```
Add a comment explaining why: fruits only spawn on grass, and stones never
turn to grass mid-episode.

**Option B:** keep the check as a runtime invariant guard, but convert to an
`assert` so the dead code is semantically flagged:
```python
for pos in new_fruit_positions & old_positions:
    assert self.ca_map is None or self.ca_map[pos] == 1, (
        f"Fruit at stone cell {pos} violates spawn invariant"
    )
    self.food_growth[pos] = min(1.0, self.food_growth[pos] + food_growth_rate)
```

### Recommendation
Option A. The invariant is already enforced at spawn time, and `assert` in a
hot loop is overhead for a condition that cannot fail without unrelated
upstream breakage.

### Files
- `tr_lbf_addon/game_runner.py` — `GameRunner.step` (3 lines → 2 lines)

### Tests
No new tests. The existing food-growth tests in
`test_lbf_gym.py::TestFoodGrowth` (or wherever they live) must continue to
pass.

### Risks
- Zero behavioral change under the spawn-invariant contract.

---

## Rollout Plan

### Sequencing
1. **Issue 3** (dead-agent filter) — eliminates phantom helpers, simplifies
   repros for #2 and #4.
2. **Issue 2** (unreachable-fruits filter) — catches the remaining
   "dead body blocks only slot" stuck case.
3. **Issue 1** (empty known_fruits guard) — trivial safety net.
4. **Issue 4** (walking-agents don't clear target) — efficiency + less churn.
5. **Issue 5** — deferred; see `plan-learned-reprediction-gate.md`. Nothing to
   implement in this plan.
6. **Issue 6** (record_ground_truth refactor) — performance polish.
7. **Issue 7** (remove unreachable ca_map check) — tidy-up.

### Per-issue acceptance checklist
For each issue, in order:
- [ ] Implement fix with docstring for any new function/method
- [ ] Add targeted tests (only where coverage gap exists)
- [ ] Run `pytest tr_lbf_addon/tests/` locally — no regressions
- [ ] Run full `pytest` suite — pre-existing `lbforaging/tests/test_env.py`
      `OrderEnf...` failures remain, no new failures
- [ ] Update `README.md` "Agent Behavior Fixes" section with a one-line
      summary of the fix
- [ ] Update `memory/project_state.md` with new test counts and short
      description

### Test-count projection
- Current: 186 passing in addon tests
- After all issues: ~186 + 3 (Issue 1) + 3 (Issue 2) + 4 (Issue 3) + 3
  (Issue 4) + 0 (Issue 5 deferred) + 1 (Issue 6) + 0 (Issue 7) = **~200 passing**

### Documentation updates
- **README.md** — add a bullet per fix under "Agent Behavior Fixes"
- **memory/project_state.md** — bump test counts with a short description of
  each fix shipped

### What we are deliberately NOT doing
- Not merging `ca_map` into `full_info_field` (user decision: keep as-is)
- Not implementing a failed-LOAD hardcoded-threshold fix for Issue 5 (rejected;
  the learned-re-prediction-gate plan addresses the root cause architecturally)
- Not touching `lbforaging/` upstream — all fixes live in `tr_lbf_addon/`

---

## Summary of file changes

| File | Issues | Approx lines touched |
|------|--------|----------------------|
| `tr_lbf_addon/lbf_elements.py` | 1, 2, 3, 4 | ~25 |
| `tr_lbf_addon/lbf_gym.py` | 6 | ~10 |
| `tr_lbf_addon/game_runner.py` | 7 | ~2 |
| `tr_lbf_addon/tests/test_lbf_elements.py` | 1, 2, 3, 4 | ~150 (new tests) |
| `tr_lbf_addon/tests/test_lbf_gym.py` | 6 | ~30 (optional new test) |
| `README.md` | docs | ~6 lines |
| `memory/project_state.md` | docs | ~8 lines |
