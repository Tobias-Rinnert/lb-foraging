# Plan — Learned Re-Prediction Gate

Replace the hardcoded re-prediction gating (`is_agent_on_predicted_path` + the
early-return in `choose_fruit` + `_STATIONARY_RESELECT_THRESHOLD`) with a
learned, self-calibrating gate that decides when each agent should re-run its
main prediction network versus reuse the cached result. The gate is tiny,
game-agnostic, co-evolved with the main NN, and adapts online via Thompson
sampling over its own observed accuracy.

This plan also subsumes **Issue 5** from `plan-stuck-edge-cases.md` — the
"adjacent-and-LOADing-forever with absent helper" stuck case. Its root cause
is that the main NN's belief about another agent's target never corrects
itself; a learned gate with oracle-probed updates fixes this structurally
instead of patching it with a hardcoded failed-LOAD counter.

Follow the project coding workflow throughout: descriptive naming, for-loops
over while-loops, vectorized numpy where possible, docstrings on every
new/changed function, add tests only where coverage is missing, run the full
suite at the end, update the README.

---

## 1. Motivation

### Why the gate matters more, not less, going forward
Every time an agent decides to **skip** re-prediction for one of its peers, it
avoids not a single NN forward pass but a full `#fruits × #features` loop in
`predict_agent_target`. With `N` agents and `F` fruits, total cost per
timestep is `O(N × (N−1) × F)` — already ~160 forward passes at N=5, F=8,
and scaling badly.

As the main NN architecture grows toward the long-term goal of a
general-purpose, game-agnostic predictor (entity-set input, attention over
arbitrarily many entities), each forward pass gets more expensive. The gate
becomes the single biggest lever for keeping inference tractable.

### Why the current gate is inadequate
- `is_agent_on_predicted_path` returns True whenever the peer's position is
  *anywhere* on the cached path, even if the peer is deviating via an
  overlapping segment. Predictions linger long past their expiry.
- The early-return in `choose_fruit` locks out re-prediction entirely when
  the focal agent's own target is still on the map and it isn't stationary.
- `_STATIONARY_RESELECT_THRESHOLD = 3` is a hand-picked constant with no
  principled basis.
- None of these can learn. They cannot adapt to a changing main-NN
  architecture, to different game parameters, or to individual agents whose
  NNs have drifted in different directions under evolution.

### Why no hardcoded thresholds
The user explicit preference: all tuning parameters should emerge from the
math or from evolutionary pressure. Human-picked thresholds are brittle
against the moving target of a co-evolving main NN.

---

## 2. Goals and Non-Goals

### Goals
- Replace the hardcoded gate with a learned binary classifier.
- Make the decision rule adaptive: probe more when recent decisions were
  wrong, probe less when recent decisions were right.
- Avoid any human-picked threshold. Only Bayesian priors (maximally
  uninformative) and evolved hyperparameters (in the genome).
- Stay game-agnostic: the gate's features must survive changes to the main
  NN's input schema.
- Eliminate Issue 5's stuck case structurally (the gate's probe mechanism
  provides the missing feedback signal for stale helper predictions).

### Non-Goals
- This plan does **not** build the time-series world-state input for the main
  NN. That's a separate README todo ("feed a time series of world states"),
  solving a different aspect of Issue 5's root cause. The two are
  complementary: time-series input lets the main NN *represent* "agent B has
  not moved for many steps"; the learned gate lets the system *act* on the
  resulting better predictions.
- This plan does **not** redesign the main `AgentPredictor` architecture.
  The gate sits alongside, not inside.
- This plan does **not** invoke a learned representation of the probing
  decision via replayable trajectories ("stories"). That's a separate and
  much bigger concept pointed at by the user's mind-vault project.

---

## 3. Design Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│  per (focal_agent, other_agent) pair, per timestep                    │
│                                                                       │
│  1. Extract change-features Δ since last prediction                   │
│  2. Gate network forward pass: p = P(output would change)             │
│  3. Sample p_correct ~ Beta(α, β)  — posterior over gate accuracy     │
│  4. Decide:                                                           │
│       if gate says "repredict":       run main NN (oracle label free) │
│       elif sampled_bernoulli(1 - p_correct):  probe, run main NN      │
│       else:                           skip, reuse cached prediction   │
│  5. If main NN ran: update α/β on whether gate's call was correct     │
│  6. If skipping: nothing to update; posterior unchanged               │
│                                                                       │
│  α/β are scaled by evolved forgetting factor ρ each step              │
└───────────────────────────────────────────────────────────────────────┘
```

### Key components
1. **Gate network**: tiny MLP, ~30 params, outputs a binary "should
   re-predict" probability.
2. **Change-feature extractor**: game-agnostic features comparing current
   state to state at last prediction.
3. **Thompson-sampling decision rule**: Beta(α, β) posterior per pair drives
   probing probability on skip decisions.
4. **Evolved forgetting factor ρ**: part of the agent genome; decays α/β over
   time to handle non-stationarity.
5. **Oracle-label generation**: every time the main NN runs, compare its
   output to the cached prediction to generate a training label for the gate.
6. **Training loop**: gate trains online on oracle labels, interleaved with
   the main NN's own training.

---

## 4. Detailed Components

### 4.1 Change-Feature Extractor

**Why game-agnostic features:** the gate must survive any future change to
the main NN's input schema. These features describe *changes*, not the raw
game state.

**Per (focal, other) pair, extract:**

| Feature | Description | Computation |
|---|---|---|
| `steps_since_last` | ticks since the last prediction for this pair | counter |
| `other_displacement` | L2 norm of peer's position change since last prediction | vector subtraction |
| `predicted_fruit_slots_delta` | change in `len(free_slots)` of the currently-predicted fruit since last prediction | count diff |
| `fruits_added` | number of new fruits on the map since last prediction | set diff |
| `fruits_removed` | number of fruits loaded/removed since last prediction | set diff |
| `last_confidence` | top probability from the cached prediction | already stored |
| `last_entropy` | entropy of the cached prediction distribution over fruits | compute from cached probs |

All features are scalars. Normalize each against fixed bounds from game
settings (e.g., `other_displacement / (field_size × 2)`).

**Implementation location:** new helper `_extract_gate_features(focal, other)`
in `lbf_elements.py`, called from `choose_fruit`.

### 4.2 Gate Network

**Architecture:** 2-layer MLP.
```python
class PredictionGate(nn.Module):
    """Binary classifier: should we re-run the main prediction NN for this pair?

    Inputs: game-agnostic change-features vector (shape: (7,) by default).
    Output: scalar in [0, 1] — probability that cached prediction is stale.
    """
    def __init__(self, n_features: int = 7, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
```

**Decision boundary:** the gate's raw output `p_gate` is treated as a
probability. Decision:
- `p_gate > 0.5` → gate says "repredict"
- `p_gate <= 0.5` → gate says "skip"

No margin, no additional threshold — the Thompson-sampling layer handles
uncertainty.

**Size:** 7 × 8 + 8 + 8 × 1 + 1 = ~73 params. Negligible overhead compared
to the main NN.

### 4.3 Thompson-Sampling Decision Rule

Per `(focal_agent_id, other_agent_id)` pair, maintain a Beta posterior over
"probability the gate's verdict is correct":

```
α[focal, other] — successes (times gate was observed correct)
β[focal, other] — failures (times gate was observed incorrect)
```

Initialize `α = β = 1` (uniform prior). This gives pessimistic warmup:
with no data, `Beta(1, 1)` is uniform on [0, 1], so sampled `p_correct`
averages 0.5 → probe rate ~50% → mostly oracle-labeled data flows in until
the posterior tightens.

**Decision flow inside `choose_fruit`:**
```python
p_gate = prediction_gate(change_features)         # in [0, 1]
p_correct = sample_beta(alpha[pair], beta[pair])  # Thompson sample

if p_gate > 0.5:
    # Gate says repredict. Run main NN. Label generated for free.
    run_main_nn_and_update_gate_label(...)
else:
    # Gate says skip. Probe with probability proportional to uncertainty.
    probe = bernoulli(1.0 - p_correct)
    if probe:
        run_main_nn_and_update_gate_label(...)
    else:
        use_cached_prediction(...)
```

**Updating α/β** after the main NN runs (either because gate said repredict
or because we probed):
```python
oracle_says_repredict = (argmax(fresh_probs) != argmax(cached_probs))
gate_said_repredict = (p_gate > 0.5)

if oracle_says_repredict == gate_said_repredict:
    alpha[pair] += 1     # gate was right
else:
    beta[pair] += 1      # gate was wrong
```

**Why this matches "thinks more when last predictions were wrong":** a run
of wrong calls increments β → Beta posterior shifts toward 0 → sampled
`p_correct` decreases → `1 - p_correct` (probe probability) increases →
more probes in the next N steps → more labels → gate retrains on the
mistakes → accuracy recovers → α catches up. Fully emergent dynamics.

### 4.4 Evolved Forgetting Factor ρ

The main NN is non-stationary:
- Online learning within an episode shifts its weights
- Reproduction between episodes creates child NNs with different parameters
- Architecture mutation changes input/output dimensions

Old α/β counts describe a past version of the main NN. To track drift:
```python
alpha[pair] = rho * alpha[pair] + correct_indicator
beta[pair]  = rho * beta[pair]  + incorrect_indicator
```
where `rho ∈ (0, 1)` is the **per-agent forgetting factor**, stored in the
genome alongside `embedding_dim` and `decision_hidden`.

**Why evolved:**
- Agent with `ρ` too low → posterior never tightens → probes too often →
  slow → low fitness
- Agent with `ρ` too high → stale counts → over-trusts gate after drift →
  skips wrongly → fewer loads → low fitness
- Selection converges `ρ` to the natural drift rate of each agent's own
  main NN

**Initialization:** first-generation agents start with `ρ = 0.99`
(effective memory ~100 steps). Mutation applies Gaussian noise in log-space
so `ρ` stays in `(0, 1)`.

**Genome additions** in `neuroevolution.py`:
```python
@dataclass
class AgentGenome:
    ...
    forgetting_factor: float = 0.99
    gate_model: PredictionGate | None = None
    gate_optimizer: torch.optim.Optimizer | None = None
```

### 4.5 Gate Training Loop

**Where labels come from:**
Every step the main NN runs (gate said repredict OR probe), we have both
the fresh and cached predictions → oracle label is free. Store the
`(features, label)` pair.

**When to train:**
Match the main NN's training cadence — inside `agent.learn()`. After the
main NN trains on its labeled predictions, also train the gate on its
labeled features:
```python
def learn_gate(self) -> list[float]:
    """Train the prediction gate on oracle-labeled features from this step.

    Oracle labels are generated free for every step where the main NN ran
    (either because the gate voted repredict, or because a Thompson-sampling
    probe triggered). Uses BCE loss with class reweighting to prioritize
    reducing false negatives (gate wrongly skipped a real change).

    Returns:
        list[float]: BCE loss value for each training sample
    """
    ...
```

**Loss:** binary cross-entropy with positive-class reweighting. False
negatives (gate says skip, oracle says repredict) cause stale predictions,
which cause stuck agents; false positives just waste compute. Weight
positive class ~5–10× higher.

**Forgetting for gate weights:** none. The gate is supervised on
per-step oracle labels; if the main NN drifts, the gate retrains on new
labels automatically.

### 4.6 What Stays Hardcoded, and Why

Even with the learned gate, keep two architectural invariants:

1. **`force_reselect=True` when any fruit was loaded.** This is not a gating
   threshold — it's a signal that the environment changed significantly
   (one of our conditioning inputs became invalid). Always repredict in this
   step so the gate has fresh oracle labels to learn from. Without this,
   post-load training data would be sparse.

2. **Stationary-recovery safety net.** Keep `_STATIONARY_RESELECT_THRESHOLD`
   as a backstop only — if the gate has catastrophically miscalibrated and
   an agent hasn't moved for 10+ steps, force a reselect regardless of what
   the gate says. This is a "the learned system failed, fall back to
   something basic" circuit breaker. It should fire approximately never
   once the gate is trained.

---

## 5. Integration Points

### 5.1 `lbf_elements.py` — `Agent`

New fields in `Agent.__init__`:
```python
self.prediction_gate: PredictionGate | None = None
self.gate_optimizer: torch.optim.Optimizer | None = None
self.forgetting_factor: float = 0.99
self.gate_beta_counts: dict[int, tuple[float, float]] = {}  # other_id → (α, β)
self.last_prediction_features: dict[int, dict] = {}  # other_id → cached features & probs
self.gate_training_buffer: list[tuple[np.ndarray, float]] = []  # (features, label)
```

New methods:
- `_extract_gate_features(other_agent_id) -> np.ndarray`
- `_should_repredict(other_agent_id) -> tuple[bool, float, float]`
  returns `(should_run_main_nn, p_gate, p_correct_sampled)`
- `_update_gate_posterior(other_agent_id, gate_decision, oracle_decision)`
- `learn_gate() -> list[float]`

Modified methods:
- `choose_fruit`: replace the `is_agent_on_predicted_path` branch with the
  gate-driven decision
- `learn`: call `learn_gate` after main-NN training

Removed (with the gate deployed):
- `is_agent_on_predicted_path` — no longer the gating mechanism. Keep it in
  the file for backward-compat during rollout, but it stops being called.

### 5.2 `lbf_gym.py` — `LBF_GYM.update_agents`

No structural change. The per-step forgetting update happens inside
`choose_fruit` naturally.

### 5.3 `neuroevolution.py` — `AgentGenome`, `reproduce`, `mutate_*`

- Add `forgetting_factor`, `gate_model`, `gate_optimizer` to the genome
- Mutate `forgetting_factor` with Gaussian noise in log-odds space
- Transfer gate weights from parent to child in `reproduce`, similar to how
  main NN weights are transferred
- Gate's architecture never mutates (always the same 2-layer MLP) — only its
  weights evolve

### 5.4 `game_runner.py`

- On first-generation init, construct each agent's gate with random weights
- Persist gate in the saved genome file alongside the main NN

### 5.5 Tests

New module: `tr_lbf_addon/tests/test_prediction_gate.py`

Coverage:
1. `test_gate_forward_pass_produces_probability` — output in [0, 1]
2. `test_gate_features_extracted_from_agent_state` — feature vector shape
   and values
3. `test_beta_posterior_initialized_uniform` — `α = β = 1` at start
4. `test_beta_update_on_correct_gate_call` — α incremented
5. `test_beta_update_on_incorrect_gate_call` — β incremented
6. `test_forgetting_factor_decays_counts` — after multiple steps with
   ρ=0.9, old counts fade
7. `test_thompson_sampling_probes_on_uncertainty` — with β=10, α=1, probe
   rate is high
8. `test_thompson_sampling_rarely_probes_when_confident` — with α=100, β=1,
   probe rate is near zero
9. `test_force_reselect_overrides_gate` — when any fruit loaded, main NN
   runs regardless
10. `test_stationary_safety_net_overrides_gate` — 10+ stationary steps
    forces repredict regardless

Extended tests in existing modules:
- `test_lbf_elements.py::TestChooseFruit` — the gate runs in place of the
  old early-return; update mocks to inject a dummy gate
- `test_neuroevolution.py` — add gate weight transfer tests

---

## 6. Phased Rollout

### Phase 0 — Prerequisite
Complete `plan-stuck-edge-cases.md` fixes (Issues 1–4, 6, 7). Those
eliminate the dominant stuck scenarios without needing the gate. The gate
then targets the remaining gap cleanly.

### Phase 1 — Scaffolding
- Add `PredictionGate` class in `neuroevolution.py`
- Add gate fields to `AgentGenome`
- Add gate fields and helper methods to `Agent`
- **Do not yet wire into `choose_fruit`.** Keep old gating active.
- Tests for the new components in isolation.

Acceptance: gate exists, is testable in isolation, full suite still green
with old gating in place.

### Phase 2 — Shadow Mode
- Wire the gate into `choose_fruit` in **shadow mode**: both old and new
  gating run; old gating's decision is used; new gating's decision is
  logged.
- Collect oracle labels and train the gate.
- Monitor in metrics: gate's shadow accuracy vs. old gating's accuracy.

Acceptance: gate learns, shadow metrics show it converging to non-trivial
accuracy (say >70% after one full episode).

### Phase 3 — Cutover
- Switch `choose_fruit` to use the new gate's decision.
- Old gating code path removed or disabled behind a flag.
- Safety net (stationary recovery) stays wired.
- Add gate to the saved-genome file.

Acceptance: no regression in episode-end food-eaten totals across 5+
episodes.

### Phase 4 — Evolution Integration
- Enable mutation of `forgetting_factor`
- Enable gate weight transfer in `reproduce`
- Run long evolutionary experiments; confirm `ρ` converges to a
  non-degenerate value

Acceptance: across 20+ generations, mean `ρ` stabilizes around a value
different from the initial 0.99 (proving selection is doing something
real).

### Phase 5 — Documentation
- Update README "Changes to LBF" / "Agent Behavior" sections
- Add gate architecture diagram
- Update `memory/project_state.md`
- Remove deprecated `is_agent_on_predicted_path` from the file once it's
  fully unused

---

## 7. Risks and Open Questions

### Cold start
Before the gate has seen any data, its outputs are random. During Phase 1
and early Phase 2 this would cause wild decisions. Mitigation: pessimistic
prior (α=β=1 already) plus in shadow mode the old gating is still
authoritative. In Phase 3, accept ~1 episode of noisy behavior as the gate
converges.

### Feature drift across game-settings changes
If `field_size` changes, the normalization bound for `other_displacement`
changes, so learned gate weights may not transfer cleanly. Store the
normalization bounds with the genome and retrain the gate if they change.

### Interaction with forced-reselect and safety-net
Both bypass the gate. If they fire constantly, the gate never gets to make
real decisions, and its posterior stays at the prior. Mitigation: count how
often they fire; if >30% of steps, something upstream is broken —
investigate.

### Thompson sampling variance
The Beta posterior's sample variance at α=β=1 is ~0.083. This means probe
decisions are noisy early on. Could add a small "exploration floor"
(probe at least p_min per step regardless), but that's a hardcoded
threshold. Better: accept the noise; it's the cost of warmup.

### The "stories" alternative (out of scope, noted)
A future, much bigger project would replace (α, β) with replayable
trajectory data — agents pass *episodes* to their children, not just
posterior moments. The current design compresses experience into two
numbers per pair; a story-based design would preserve generative structure.
Flagged as an upgrade path.

### Gate training cost
The gate's `learn_gate` call runs per step, same as `learn()` for the main
NN. With ~30 params and small batches (labels from this step only), cost
is negligible. But: if the main NN is skipped often (gate works), we get
few labels per step. This is *good* for compute but *bad* for gate
training. Acceptable trade-off: once the gate is well-calibrated, it
shouldn't need much more data.

---

## 8. Summary

| Component | Purpose | Hand-picked constants |
|---|---|---|
| Change-feature extractor | Game-agnostic Δ between now and last prediction | None (normalization uses genome-stored bounds) |
| Gate network | Binary classifier for "is cached prediction stale" | Architecture size (2-layer, 8-hidden) — structural, not a threshold |
| Beta posterior | Tracks gate's observed accuracy per pair | α₀ = β₀ = 1 (uniform prior — maximally uninformative) |
| Thompson sampling | Adaptive probing proportional to uncertainty | None |
| Forgetting factor ρ | Tracks non-stationary gate accuracy | None (ρ is in the genome, evolves) |
| `force_reselect` on fruit load | Fresh label after big env change | Not a gate threshold; architectural invariant |
| Stationary-recovery safety net | Circuit breaker for catastrophic gate failure | `_STATIONARY_RESELECT_THRESHOLD` (safety only; should approximately never fire) |

**Estimated code size:**
- `lbf_elements.py` — ~80 new lines (helper methods, gate call, buffer)
- `neuroevolution.py` — ~50 new lines (PredictionGate class, genome fields, mutation)
- `game_runner.py` — ~15 new lines (gate init, save/load)
- `tests/test_prediction_gate.py` — ~200 lines (new test module)
- Existing test updates — ~50 lines

**Estimated test count:** ~10 new tests in the new module, plus ~5 updates
to existing tests → roughly +15 over the stuck-edge-cases plan's projection.

---

## 9. Relationship to Other Plans

- **`plan-stuck-edge-cases.md`** — prerequisite. Fixes structural bugs and
  filters so the remaining stuck cases are all belief-mismatch cases that
  the gate addresses. Issue 5 in that plan is now redirected here.
- **README todo: time-series NN input** — complementary. Gives the main NN
  the *ability* to represent "agent has been idle for N steps"; the gate
  gives the *trigger* to act on updated beliefs. Either alone is partial;
  both together is the full architectural fix for the stuck-cooperative
  case.
- **Future: "stories" design (mind-vault project)** — replaces α/β with
  replayable trajectories. Out of scope for this plan but pointed at as the
  next architectural step.
