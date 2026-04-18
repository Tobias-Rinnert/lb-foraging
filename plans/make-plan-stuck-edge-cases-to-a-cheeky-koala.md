# Plan — Stuck-Agent & Robustness Edge Cases (Haiku-Executable)

## Context

Six fixes for stuck-agent edge cases and robustness. Every change below is mechanical: exact file path, exact `old_string` to search for, exact `new_string` to replace it with, exact test code to append, and exact command to verify the step.

The fixes eliminate classes of stuck-agent behavior (empty `known_fruits` crash, unreachable fruits selected, dead agents pollute cooperation, walking agents clear target unnecessarily), a performance nit in `record_ground_truth`, and an unreachable defensive check in `food_growth`.

**Execution order (do NOT deviate):** Issue 3 → Issue 2 → Issue 1 → Issue 4 → Issue 6 → Issue 7. Issue 3 first because it eliminates repro cases for 2/4.

**Deferred / not covered here:** the "adjacent-and-LOADing-forever with absent helper" case (formerly Issue 5). Root cause is architectural (NN has no negative-feedback signal for failed cooperation; re-prediction gate is hand-coded) and is addressed in [plan-learned-reprediction-gate.md](plan-learned-reprediction-gate.md). Interim mitigation: Issue 2 drops fruits with empty `free_slots`, Issue 3 drops dead helpers, and `_STATIONARY_RESELECT_THRESHOLD = 3` acts as a safety net.

---

## Prerequisites (run once at session start)

```bash
cd C:/Users/bobis/development/lb-foraging
python -m pytest tr_lbf_addon/tests/ -q
```

Expected: **186 passed** (the 3 `OrderEnf...` failures in `lbforaging/tests/test_env.py` are pre-existing and unrelated; do not try to fix them).

**Coding workflow rules (apply to every change):**
- Descriptive names, for-loops over while-loops, vectorize with numpy where natural.
- Every new method/function gets a docstring. Every new test gets a one-line docstring.
- Edit tests only when coverage is missing for the change you make.
- After each issue: run the targeted test file, then the full addon suite.
- After ALL issues: update `README.md` and `memory/project_state.md`.

---

# Issue 3 — Filter Dead Agents from `known_agents` (DO FIRST)

Dead agents leak into coop-level sums, NN predictions, and expected-reward candidate helpers. Filter them at the source (`process_agent_infos`) and also purge stale predictions in `choose_fruit`.

### Step 3.1 — Edit `process_agent_infos`

**File:** `tr_lbf_addon/lbf_elements.py` (method at ~line 193)

Use the Edit tool with:

**old_string:**
```
        self.known_agents = [{"id": agent.id,
                                   "position": agent.position,
                                   "level": agent.level,
                                   "position_history": agent.position_history[-self.memory_size:],
                                   "last_action": agent.last_action,
                                   "is_loading": agent.is_loading
                                   } for agent in agents if agent.id != self.id]
```

**new_string:**
```
        self.known_agents = [{"id": agent.id,
                                   "position": agent.position,
                                   "level": agent.level,
                                   "position_history": agent.position_history[-self.memory_size:],
                                   "last_action": agent.last_action,
                                   "is_loading": agent.is_loading
                                   } for agent in agents if agent.id != self.id and agent.is_alive]
```

### Step 3.2 — Add stale-prediction purge in `choose_fruit`

**File:** `tr_lbf_addon/lbf_elements.py` (inside `choose_fruit`, right after the "Invalidate predictions for fruits that are no longer on the map" block, ~line 381)

**old_string:**
```
        # Invalidate predictions for fruits that are no longer on the map
        for agent_id in list(self.predicted_targets.keys()):
            predicted_fruit = self.predicted_targets[agent_id]
            if not np.any(np.all(predicted_fruit.position == fruit_positions, axis=1)):
                del self.predicted_targets[agent_id]
                del self.predicted_paths[agent_id]
                del self.prediction_round[agent_id]

        # For each other agent: skip NN call if still on predicted path, else re-predict
```

**new_string:**
```
        # Invalidate predictions for fruits that are no longer on the map
        for agent_id in list(self.predicted_targets.keys()):
            predicted_fruit = self.predicted_targets[agent_id]
            if not np.any(np.all(predicted_fruit.position == fruit_positions, axis=1)):
                del self.predicted_targets[agent_id]
                del self.predicted_paths[agent_id]
                del self.prediction_round[agent_id]

        # Drop predictions pinned to agents that are no longer alive
        live_agent_ids = {a["id"] for a in self.known_agents}
        for agent_id in list(self.predicted_targets.keys()):
            if agent_id not in live_agent_ids:
                del self.predicted_targets[agent_id]
                del self.predicted_paths[agent_id]
                del self.prediction_round[agent_id]

        # For each other agent: skip NN call if still on predicted path, else re-predict
```

### Step 3.3 — Add tests

**File:** `tr_lbf_addon/tests/test_lbf_elements.py`

First read the file to confirm existing fixture names (`agent`, `simple_grid`, `fruit_with_free_slots`) and the `Agent` constructor signature. Then append the following class at the end of the file (replace fixture usage if a better-matching fixture exists in that file):

```python
class TestDeadAgentFiltering:
    """Dead agents must be excluded from cognition subsystems (issue 3)."""

    def test_dead_agent_excluded_from_known_agents(self, simple_grid):
        """process_agent_infos drops agents whose is_alive is False."""
        focal = Agent(id=0, position=np.array([0, 0]), level=1)
        focal.path_finding_grid = simple_grid
        alive = Agent(id=1, position=np.array([1, 1]), level=2)
        dead = Agent(id=2, position=np.array([2, 2]), level=3)
        dead.is_alive = False
        focal.process_agent_infos([focal, alive, dead])
        ids = {a["id"] for a in focal.known_agents}
        assert ids == {1}

    def test_dead_agent_level_not_in_coop_sums(self, simple_grid):
        """After filtering, coop-level sums must not include dead agents' levels."""
        focal = Agent(id=0, position=np.array([0, 0]), level=1)
        focal.path_finding_grid = simple_grid
        alive = Agent(id=1, position=np.array([1, 1]), level=2)
        dead = Agent(id=2, position=np.array([2, 2]), level=4)
        dead.is_alive = False
        focal.process_agent_infos([focal, alive, dead])
        other_levels = [a["level"] for a in focal.known_agents]
        sums = focal.get_possible_coop_level_sums(other_levels)
        assert 5 not in sums  # 1 + 4 (dead) would be 5
        assert 7 not in sums  # 1 + 2 + 4 would be 7

    def test_dead_agent_predictions_invalidated_on_choose_fruit(self, simple_grid):
        """predicted_targets/paths/rounds for a dead agent are purged by choose_fruit."""
        focal = Agent(id=0, position=np.array([0, 0]), level=2)
        focal.path_finding_grid = simple_grid
        alive = Agent(id=1, position=np.array([1, 1]), level=1)
        dead = Agent(id=2, position=np.array([2, 2]), level=1)
        dead.is_alive = False
        focal.process_agent_infos([focal, alive, dead])

        stale_fruit = Fruit(position=np.array([3, 3]), level=1, free_slots=[np.array([3, 4])])
        focal.known_fruits = [stale_fruit]
        focal.predicted_targets = {2: stale_fruit}
        focal.predicted_paths = {2: np.array([[2, 2], [3, 3]])}
        focal.prediction_round = {2: 0}

        focal.choose_fruit(force_reselect=True)

        assert 2 not in focal.predicted_targets
        assert 2 not in focal.predicted_paths
        assert 2 not in focal.prediction_round
```

If `Agent` or `Fruit` aren't imported at the top of `test_lbf_elements.py`, add the appropriate `from tr_lbf_addon.lbf_elements import Agent, Fruit` (check existing imports first — do not duplicate).

### Step 3.4 — Verify

```bash
python -m pytest tr_lbf_addon/tests/test_lbf_elements.py -q
```

Expected: all prior tests + 3 new tests pass. If `get_possible_coop_level_sums` gives unexpected values, read the method body and adjust the literal `5`/`7` asserts to match the actual behavior rather than weakening the test.

---

# Issue 2 — Unreachable Fruits Stay in `feasible_fruits`

A fruit whose `free_slots == []` is still feasible by level; `choose_fruit` picks it, `choose_next_action` can't path, target clears, same fruit re-picked next step. Add a `fruit.free_slots` conjunction to the feasibility filter.

### Step 2.1 — Edit `choose_fruit` feasibility filter

**File:** `tr_lbf_addon/lbf_elements.py` (inside `choose_fruit`, ~line 370)

**old_string:**
```
        feasible_fruits = [
            fruit for fruit in self.known_fruits
            if fruit.level <= max_achievable
        ]
```

**new_string:**
```
        feasible_fruits = [
            fruit for fruit in self.known_fruits
            if fruit.level <= max_achievable and fruit.free_slots
        ]
```

### Step 2.2 — Add tests

Append to `TestChooseFruit` in `tr_lbf_addon/tests/test_lbf_elements.py`. First read the file to locate the class. Then add (inside the class, using existing fixtures):

```python
    def test_fruit_without_free_slots_excluded_from_feasible(self, agent):
        """choose_fruit ignores fruits whose free_slots list is empty."""
        blocked = Fruit(position=np.array([1, 1]), level=1, free_slots=[])
        reachable = Fruit(position=np.array([3, 3]), level=1,
                          free_slots=[np.array([3, 4])])
        agent.known_fruits = [blocked, reachable]
        agent.known_agents = []
        agent.choose_fruit(force_reselect=True)
        assert agent.target is reachable

    def test_all_fruits_without_free_slots_yields_none_target(self, agent):
        """When every known fruit is blocked, target becomes None."""
        blocked_a = Fruit(position=np.array([1, 1]), level=1, free_slots=[])
        blocked_b = Fruit(position=np.array([3, 3]), level=1, free_slots=[])
        agent.known_fruits = [blocked_a, blocked_b]
        agent.known_agents = []
        agent.choose_fruit(force_reselect=True)
        assert agent.target is None

    def test_previously_blocked_fruit_selectable_after_slots_free_up(self, agent):
        """A fruit whose slots reopen must be eligible on the next choose_fruit call."""
        fruit = Fruit(position=np.array([2, 2]), level=1, free_slots=[])
        agent.known_fruits = [fruit]
        agent.known_agents = []
        agent.choose_fruit(force_reselect=True)
        assert agent.target is None

        fruit.free_slots = [np.array([2, 3])]
        agent.choose_fruit(force_reselect=True)
        assert agent.target is fruit
```

### Step 2.3 — Verify

```bash
python -m pytest tr_lbf_addon/tests/test_lbf_elements.py::TestChooseFruit -q
```

All previous `TestChooseFruit` cases must still pass. If any regresses, it was probably asserting the old (wrong) behavior of selecting a blocked fruit — revise that legacy assertion to the new correct behavior, do not revert the fix.

---

# Issue 1 — `choose_fruit` Crashes on Empty `known_fruits`

`np.array([])` is 1-D; `np.all(..., axis=1)` raises `AxisError`. Guard at the top.

### Step 1.1 — Add early return

**File:** `tr_lbf_addon/lbf_elements.py` (inside `choose_fruit`, ~line 354, right after the docstring closes)

**old_string:**
```
            None (sets self.target side effect)
        """
        fruit_positions = np.array([fruit.position for fruit in self.known_fruits])
```

**new_string:**
```
            None (sets self.target side effect)
        """
        if not self.known_fruits:
            self.target = None
            return

        fruit_positions = np.array([fruit.position for fruit in self.known_fruits])
```

### Step 1.2 — Add tests

Append inside `TestChooseFruit`:

```python
    def test_empty_known_fruits_clears_target_without_crash(self, agent):
        """Empty known_fruits must clear the target and not raise."""
        stale = Fruit(position=np.array([2, 2]), level=1, free_slots=[np.array([2, 3])])
        agent.target = stale
        agent.known_fruits = []
        agent.known_agents = []
        agent.choose_fruit()
        assert agent.target is None

    def test_empty_known_fruits_with_force_reselect(self, agent):
        """Force-reselect path also handles empty known_fruits cleanly."""
        stale = Fruit(position=np.array([2, 2]), level=1, free_slots=[np.array([2, 3])])
        agent.target = stale
        agent.known_fruits = []
        agent.known_agents = []
        agent.choose_fruit(force_reselect=True)
        assert agent.target is None
```

### Step 1.3 — Verify

```bash
python -m pytest tr_lbf_addon/tests/test_lbf_elements.py::TestChooseFruit -q
```

---

# Issue 4 — Walking Agents on Slots Should Not Clear Target

`choose_next_action` currently treats every other agent's position as an obstacle to `free_slots`. Only **loading** agents are true obstacles; walking agents move. Filter by `is_loading` only.

### Step 4.1 — Edit `other_positions` filter in `choose_next_action`

**File:** `tr_lbf_addon/lbf_elements.py` (inside `choose_next_action`, ~line 248)

**old_string:**
```
        # Exclude slots already occupied by other agents so agents spread out
        other_positions = {tuple(a["position"]) for a in (self.known_agents or [])}
        available_slots = [s for s in self.target.free_slots if tuple(s) not in other_positions]
        if not available_slots:
            self.target = None  # all slots taken, try a different fruit next step
            return np.int64(0)
```

**new_string:**
```
        # Exclude slots held by LOADING agents only — they will not move out of the way.
        # Walking agents are allowed; A* routes through them (path_finding_grid only
        # marks loading agents as obstacles) and env collision handles the tick itself.
        loading_positions = {
            tuple(a["position"])
            for a in (self.known_agents or [])
            if a.get("is_loading")
        }
        available_slots = [s for s in self.target.free_slots if tuple(s) not in loading_positions]
        if not available_slots:
            self.target = None  # every slot held by a loading agent, try another fruit next step
            return np.int64(0)
```

### Step 4.2 — Add tests

Append inside `TestChooseNextAction` in `tr_lbf_addon/tests/test_lbf_elements.py`:

```python
    def test_walking_agent_on_slot_does_not_clear_target(self, agent, fruit_with_free_slots):
        """A walking (non-loading) agent occupying the closest slot keeps target set."""
        agent.position = np.array([0, 0])
        agent.target = fruit_with_free_slots
        agent.known_agents = [{
            "id": 1,
            "position": np.array([2, 3]),  # closest slot of fruit at (2,2)
            "level": 1,
            "position_history": [],
            "last_action": 0,
            "is_loading": False,
        }]
        action = agent.choose_next_action()
        assert agent.target is fruit_with_free_slots
        assert action != np.int64(0)  # a movement, not idle

    def test_loading_agent_on_slot_excluded_from_available(self, agent, fruit_with_free_slots):
        """A loading agent on the closest slot routes the focal agent to a different slot."""
        agent.position = np.array([0, 0])
        agent.target = fruit_with_free_slots
        agent.known_agents = [{
            "id": 1,
            "position": np.array([2, 3]),
            "level": 1,
            "position_history": [],
            "last_action": 0,
            "is_loading": True,
        }]
        action = agent.choose_next_action()
        assert agent.target is fruit_with_free_slots
        # path_goal must not be the blocked slot
        assert not np.array_equal(agent.path_goal, np.array([2, 3]))

    def test_all_slots_held_by_loading_agents_clears_target(self, agent, fruit_with_free_slots):
        """If every free_slot is occupied by a loading agent, target clears."""
        agent.position = np.array([0, 0])
        agent.target = fruit_with_free_slots
        agent.known_agents = [
            {"id": i + 1, "position": np.array(slot), "level": 1,
             "position_history": [], "last_action": 0, "is_loading": True}
            for i, slot in enumerate(fruit_with_free_slots.free_slots)
        ]
        action = agent.choose_next_action()
        assert agent.target is None
        assert action == np.int64(0)
```

### Step 4.3 — Verify

```bash
python -m pytest tr_lbf_addon/tests/test_lbf_elements.py::TestChooseNextAction -q
```

Any pre-existing test that asserted the old walking-agent-clears-target behavior must be updated — it was asserting the bug. Only weaken assertions, don't revert the fix.

---

# Issue 6 — `record_ground_truth` O(agents × predictions × agents) Refactor

Replace the `next(...)` scan with a dict lookup and vectorize the adjacency check.

### Step 6.1 — Rewrite the method body

**File:** `tr_lbf_addon/lbf_gym.py` (method at line 317)

**old_string:**
```
        current_positions = [f.position for f in self.fruits]

        for prev_fruit in previous_fruits:
            fruit_still_present = any(
                np.array_equal(prev_fruit.position, curr_pos) for curr_pos in current_positions
            )
            if fruit_still_present:
                continue

            self.any_fruit_loaded = True

            adjacent_slots = [
                prev_fruit.position + np.array([0, 1]),
                prev_fruit.position + np.array([0, -1]),
                prev_fruit.position + np.array([1, 0]),
                prev_fruit.position + np.array([-1, 0]),
            ]

            for agent in self.agents:
                for prediction in agent.predictions:
                    if prediction["ground_truth"] is not None:
                        continue
                    if not np.array_equal(prediction["fruit_pos"], prev_fruit.position):
                        continue
                    predicted_agent = next(
                        (a for a in self.agents if a.id == prediction["agent_id"]), None
                    )
                    if predicted_agent is None:
                        continue
                    was_adjacent = any(
                        np.array_equal(predicted_agent.position, slot) for slot in adjacent_slots
                    )
                    prediction["ground_truth"] = 1.0 if was_adjacent else 0.0
```

**new_string:**
```
        current_positions = [f.position for f in self.fruits]

        # Precomputed once per call: O(agents) dict replaces the per-prediction linear scan.
        agent_positions_by_id = {a.id: a.position for a in self.agents}

        for prev_fruit in previous_fruits:
            fruit_still_present = any(
                np.array_equal(prev_fruit.position, curr_pos) for curr_pos in current_positions
            )
            if fruit_still_present:
                continue

            self.any_fruit_loaded = True

            # Vectorized adjacency: shape (4, 2) so we can broadcast-compare per prediction.
            adjacent_slots_arr = np.array([
                prev_fruit.position + np.array([0, 1]),
                prev_fruit.position + np.array([0, -1]),
                prev_fruit.position + np.array([1, 0]),
                prev_fruit.position + np.array([-1, 0]),
            ])

            for agent in self.agents:
                for prediction in agent.predictions:
                    if prediction["ground_truth"] is not None:
                        continue
                    if not np.array_equal(prediction["fruit_pos"], prev_fruit.position):
                        continue
                    predicted_position = agent_positions_by_id.get(prediction["agent_id"])
                    if predicted_position is None:
                        continue
                    was_adjacent = bool(
                        np.any(np.all(adjacent_slots_arr == predicted_position, axis=1))
                    )
                    prediction["ground_truth"] = 1.0 if was_adjacent else 0.0
```

### Step 6.2 — Verify (no new tests — existing `TestRecordGroundTruth` covers this)

```bash
python -m pytest tr_lbf_addon/tests/test_lbf_gym.py::TestRecordGroundTruth -q
```

Expected: all 4 existing `TestRecordGroundTruth` cases pass unchanged. Observable behavior is identical; only call cost changes.

---

# Issue 7 — Remove Unreachable `ca_map` Check in `food_growth`

Fruits cannot spawn on stone cells (enforced by `_is_empty_location` in the vendored env). The runtime check in `game_runner.py` is dead code.

### Step 7.1 — Remove the guard

**File:** `tr_lbf_addon/game_runner.py` (lines ~231-234)

**old_string:**
```
        food_growth_rate = self.params.get("food_growth_rate", 0.005)
        for pos in new_fruit_positions & old_positions:
            if self.ca_map is None or self.ca_map[pos] == 1:  # grass cell
                self.food_growth[pos] = min(1.0, self.food_growth[pos] + food_growth_rate)
```

**new_string:**
```
        # Fruits can only spawn on grass (enforced by environment._is_empty_location
        # against ca_map at spawn time) and stones never turn to grass mid-episode,
        # so we can grow every tracked fruit unconditionally.
        food_growth_rate = self.params.get("food_growth_rate", 0.005)
        for pos in new_fruit_positions & old_positions:
            self.food_growth[pos] = min(1.0, self.food_growth[pos] + food_growth_rate)
```

### Step 7.2 — Verify

```bash
python -m pytest tr_lbf_addon/tests/test_lbf_gym.py -q
```

All food-growth tests must remain green.

---

# Final Verification

1. **Full addon suite:**
   ```bash
   python -m pytest tr_lbf_addon/tests/ -q
   ```
   Expected: **~200 passed** (186 prior + 3 Issue1 + 3 Issue2 + 3 Issue3 + 3 Issue4 = 198 minimum; higher if any subsumed coverage adds). No new failures.

2. **Full repo suite (sanity):**
   ```bash
   python -m pytest -q
   ```
   Expected: same count plus the 3 pre-existing `OrderEnf...` failures in `lbforaging/tests/test_env.py`. No **new** failures outside that list.

3. **Smoke-run the game** (manual, one short episode; skip if no display is available):
   ```bash
   python run_the_game.py
   ```
   Watch for: no `AxisError` traceback; agents do not park on unreachable fruits for more than `_STATIONARY_RESELECT_THRESHOLD` steps; dead agents are not waited on.

---

# Documentation Updates (do LAST, after all tests green)

### README.md

1. Read the file and find the `## Changes to LBF` section (line ~42).
2. If a subsection titled `### Agent Behavior Fixes` exists immediately after it, append six bullets to that subsection. If not, insert a new subsection **immediately before** `## Changes to LBF` (so addon fixes come before vendored-env changes) with this content:

```markdown
### Agent Behavior Fixes (this patch)

- **Dead-agent filter** (`process_agent_infos`, `choose_fruit`): dead agents drop out of `known_agents` at the source, so coop-level sums, NN predictions, and expected-reward helper subsets only see live peers. Stale predictions keyed on dead agents are purged in `choose_fruit`.
- **Unreachable-fruit filter** (`choose_fruit`): feasibility filter now also requires `fruit.free_slots` to be non-empty, eliminating the flip-flop where an agent re-picks a blocked fruit every step.
- **Empty `known_fruits` guard** (`choose_fruit`): short-circuits to `self.target = None` before the `np.all(..., axis=1)` call that crashed with `AxisError` on a 1-D empty array.
- **Walking-agents-are-not-obstacles** (`choose_next_action`): only loading agents block available slots; walking agents are allowed to occupy the route and move away by arrival time.
- **`record_ground_truth` O(n²) → O(n) lookup**: precomputed `agent_positions_by_id` dict replaces a per-prediction linear scan; adjacency check vectorized via numpy broadcasting.
- **Dead `ca_map` check in `food_growth` removed**: fruits can only spawn on grass, so the runtime stone-guard was unreachable; removed with a comment documenting the invariant.
```

### memory/project_state.md

Update the existing project_state memory with a one-line bump per fix and the new test count. Use the Write tool to rewrite the file, bumping test counts and appending fix notes to the "Agent Behavior Fixes" block. If unsure about exact current content, read it first and preserve all lines you are not explicitly changing.

---

# Deliberately NOT Doing

- **Not merging `ca_map` into `full_info_field`.** Prior user decision — keep the two data structures separate.
- **Not implementing a failed-LOAD hardcoded-threshold fix** for the deferred "adjacent-and-LOADing-forever with absent helper" case. The root cause (no negative-feedback signal for failed cooperation, plus a hand-coded re-prediction gate) is addressed architecturally in [plan-learned-reprediction-gate.md](plan-learned-reprediction-gate.md). Interim mitigation is the combination of Issue 2 (drops unreachable fruits), Issue 3 (drops dead helpers), and the existing `_STATIONARY_RESELECT_THRESHOLD` safety net.
- **Not touching `lbforaging/` upstream.** All fixes live in `tr_lbf_addon/`.
- **Not adding per-issue risks/rollout narrative.** The original high-level plan with rationale was superseded; this file is the single source of implementation truth. If future context is needed, check git history of this plan file.

---

# Critical Files (quick reference)

- Source: [tr_lbf_addon/lbf_elements.py](tr_lbf_addon/lbf_elements.py) — Issues 1, 2, 3, 4
- Source: [tr_lbf_addon/lbf_gym.py](tr_lbf_addon/lbf_gym.py) — Issue 6 (`record_ground_truth` at line 317)
- Source: [tr_lbf_addon/game_runner.py](tr_lbf_addon/game_runner.py) — Issue 7 (food-growth loop at line ~231)
- Tests: [tr_lbf_addon/tests/test_lbf_elements.py](tr_lbf_addon/tests/test_lbf_elements.py) — new tests for 1, 2, 3, 4
- Tests: [tr_lbf_addon/tests/test_lbf_gym.py](tr_lbf_addon/tests/test_lbf_gym.py) — existing `TestRecordGroundTruth` covers Issue 6
- Docs: [README.md](README.md) — add `### Agent Behavior Fixes` under `## Changes to LBF`
- Memory: `C:/Users/bobis/.claude/projects/C--Users-bobis-development-lb-foraging/memory/project_state.md`

# Haiku Self-Check Before Finishing

- [ ] Every `Edit` call used a unique `old_string` (copy-pasted verbatim from current file, not this plan if the file differs — re-Read if uncertain).
- [ ] No new comments explaining WHAT the code does; only the WHY comments already shown.
- [ ] Added no new dependencies, imports, or helpers beyond what is listed here.
- [ ] Did not touch `lbforaging/` — all changes live in `tr_lbf_addon/`.
- [ ] Final `pytest tr_lbf_addon/tests/` run is green and the new test count matches the projection.
- [ ] README and project_state memory updated **last**, after tests are green.
