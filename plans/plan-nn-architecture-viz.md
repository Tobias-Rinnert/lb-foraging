# Plan — Interactive Neural Network Architecture Visualization

## Context

The project's `AgentPredictor` ([tr_lbf_addon/neuroevolution.py:18-110](tr_lbf_addon/neuroevolution.py)) is evolving: per-agent `embedding_dim` and `decision_hidden` are mutated each generation, weights transfer from parent to child, and the architecture may gain new layers/modules as the project progresses. Users currently have no way to inspect what the network is doing: they see probabilities on the board and loss curves in the metrics drawer, but the model is a black box.

**Goal:** an interactive visualization in the existing React web UI that lets the user (a) see the network as a zoomable graph (system → modules → layers → weights/activations), (b) click any element to drill in, (c) auto-adapt when the architecture changes (no hard-coded diagrams).

**Non-goals:** training-curve visualization (already covered by `MetricsPanel`), any form of live streaming during play (learning metrics cover that need), editing the architecture from the UI.

**Design decisions (confirmed by user):**

1. **Data-driven, not hand-drawn.** The viz is built from `torch.nn.Module` introspection (`named_modules()`, `named_parameters()`, `named_children()`). Any new layer type or dim change shows up automatically.
2. **React Flow for graph layout, Canvas for per-layer heatmaps, Recharts for 1D bars.** Matches the frontend's existing hybrid (Canvas renderer + Recharts charts); adds one dep (`@xyflow/react` v12).
3. **Opening the panel pauses the game and snapshots the current state.** No polling, no streaming. When the user opens the NN panel: backend force-pauses the game loop, then returns architecture + weights + a single hooked-forward-pass activation snapshot for the currently-visible tick. The user browses that frozen snapshot; clicking "Refresh snapshot" re-runs the forward pass for the current (still-paused) tick. Closing the panel leaves the pause state as-is — the user resumes manually via the existing Play button.
4. **Per-agent viz.** Each agent has its own genome/model, so the panel has an agent-selector. When reproduction happens across episodes, re-opening the panel pulls the updated architecture.
5. **Weights are pulled lazily per-layer** (not all at once) to keep payloads small when dims grow.
6. **`ARCHITECTURE_TEMPLATES` doubles as a text-based architecture overview.** Since the template is a small declarative dict mapping top-level model class name → nodes + edges, it is itself a human-readable spec of the network wiring. Useful as a single source of truth that complements the ASCII-art diagram currently in `README.md`.

Execution is staged in six phases so each phase is independently shippable and testable.

---

## Files — new and modified

### New — backend
- `tr_lbf_addon/nn_introspection.py` — pure introspection utilities (no FastAPI imports). Unit-testable in isolation.
- `web/backend/nn_routes.py` — HTTP + WebSocket handlers that call `nn_introspection`.
- `tr_lbf_addon/tests/test_nn_introspection.py` — tests for introspection.
- `web/backend/tests/test_nn_routes.py` — first backend tests in the repo.

### New — frontend
- `web/frontend/src/components/nn/NNVizPanel.tsx` — top-level panel (toggled from Toolbar, overlays like SettingsPanel).
- `web/frontend/src/components/nn/ArchitectureGraph.tsx` — React Flow graph with expand/collapse.
- `web/frontend/src/components/nn/nodes/ModuleNode.tsx` — custom React Flow node (Sequential/ModuleList/custom modules).
- `web/frontend/src/components/nn/nodes/LayerNode.tsx` — custom node for leaf layers (Linear/ReLU/Sigmoid/…).
- `web/frontend/src/components/nn/nodes/IONode.tsx` — custom node for inputs/outputs.
- `web/frontend/src/components/nn/DetailPanel.tsx` — side drawer showing the selected node's weights/activations.
- `web/frontend/src/components/nn/WeightHeatmap.tsx` — Canvas-backed heatmap (follows the `renderer.ts` style).
- `web/frontend/src/components/nn/ActivationBars.tsx` — Recharts bar chart.
- `web/frontend/src/hooks/useNNArchitecture.ts` — fetch + cache architecture for an agent_id.
- `web/frontend/src/hooks/useNNWeights.ts` — lazy weight fetch per layer.
- `web/frontend/src/types/nn.ts` — shared types.

### Modified
- `web/backend/server.py` — mount `nn_routes` router and handle the new WebSocket message types.
- `web/backend/serializer.py` — add `architecture_hash` per agent to the frame so the frontend can detect evolution without extra polling.
- `web/frontend/src/components/Toolbar.tsx` — add a **NN** toggle button next to the metrics/settings toggles.
- `web/frontend/src/App.tsx` — wire `NNVizPanel` visibility + selected agent state.
- `web/frontend/package.json` — add `@xyflow/react`.
- `README.md` — new section "Neural Network Visualization" under the main feature list.
- `memory/project_state.md` — add a block describing the viz once shipped.

---

## Data contract (shared between backend and frontend)

### `GET /api/nn/{agent_id}/architecture`

```jsonc
{
  "agent_id": 0,
  "architecture_hash": "a1b2c3…",     // stable hash of {module structure + dims}
  "embedding_dim": 8,
  "decision_hidden": 16,
  "nodes": [
    { "id": "input.fruit_level",     "kind": "input",  "shape": [1],           "label": "fruit_level" },
    { "id": "input.focal_features",  "kind": "input",  "shape": [2],           "label": "focal_features" },
    { "id": "input.others_features", "kind": "input",  "shape": [null, 2],     "label": "others_features (N, 2)" },
    { "id": "agent_encoder",         "kind": "module", "type": "Sequential",   "param_count": 24, "parent": null },
    { "id": "agent_encoder.0",       "kind": "layer",  "type": "Linear", "parent": "agent_encoder",
      "in_features": 2, "out_features": 8, "param_count": 24, "has_weight": true, "has_bias": true },
    { "id": "agent_encoder.1",       "kind": "layer",  "type": "ReLU",   "parent": "agent_encoder", "param_count": 0 },
    { "id": "attention_query",       "kind": "layer",  "type": "Linear", "parent": null,
      "in_features": 8, "out_features": 8, "param_count": 72, "has_weight": true, "has_bias": true },
    { "id": "attention",             "kind": "op",     "type": "ScaledDotProduct", "parent": null, "param_count": 0,
      "note": "bmm + softmax + bmm — derived, not a Module" },
    { "id": "decision_net",          "kind": "module", "type": "Sequential", "param_count": 193, "parent": null },
    { "id": "decision_net.0",        "kind": "layer",  "type": "Linear", "parent": "decision_net",
      "in_features": 11, "out_features": 16, "param_count": 192, "has_weight": true, "has_bias": true },
    { "id": "decision_net.1",        "kind": "layer",  "type": "ReLU",    "parent": "decision_net", "param_count": 0 },
    { "id": "decision_net.2",        "kind": "layer",  "type": "Linear",  "parent": "decision_net",
      "in_features": 16, "out_features": 1, "param_count": 17, "has_weight": true, "has_bias": true },
    { "id": "decision_net.3",        "kind": "layer",  "type": "Sigmoid", "parent": "decision_net", "param_count": 0 },
    { "id": "output.probability",    "kind": "output", "shape": [1], "label": "P(target=fruit)" }
  ],
  "edges": [
    { "from": "input.focal_features",  "to": "agent_encoder",   "tensor_shape": [2] },
    { "from": "input.others_features", "to": "agent_encoder",   "tensor_shape": [null, 2], "note": "shared weights" },
    { "from": "agent_encoder",         "to": "attention_query", "tensor_shape": [8],       "branch": "focal" },
    { "from": "agent_encoder",         "to": "attention",       "tensor_shape": [null, 8], "branch": "others" },
    { "from": "attention_query",       "to": "attention",       "tensor_shape": [8] },
    { "from": "attention",             "to": "decision_net",    "tensor_shape": [8],       "label": "context" },
    { "from": "input.fruit_level",     "to": "decision_net",    "tensor_shape": [1] },
    { "from": "input.focal_features",  "to": "decision_net",    "tensor_shape": [2] },
    { "from": "decision_net",          "to": "output.probability", "tensor_shape": [1] }
  ]
}
```

**Note on the I/O edges + `attention` op:** `AgentPredictor.forward` does `bmm` + `softmax` + `bmm` between `attention_query` and the others-embeddings. These are tensor ops, not `nn.Module` subclasses, so they don't appear in `named_modules()`. The introspection module synthesizes them from a small hand-written **semantic template** keyed on `AgentPredictor.__name__` — described below under the Future-Proofing note.

### `GET /api/nn/{agent_id}/weights/{node_id}`

```jsonc
{
  "agent_id": 0,
  "node_id": "agent_encoder.0",
  "weight": { "shape": [8, 2], "values": [[…]], "min": -0.52, "max": 0.51, "mean": 0.01 },
  "bias":   { "shape": [8],    "values": [ … ], "min": -0.12, "max": 0.09, "mean": -0.01 }
}
```

### WebSocket — open-panel handshake and snapshot

Client → server when the NN panel opens:
```jsonc
{ "type": "open_nn_panel", "agent_id": 0 }
```

Server behavior: force-pause the game loop (same code path as the existing `pause` message), then return one combined payload with architecture + initial activation snapshot:

```jsonc
{
  "type": "nn_panel_ready",
  "agent_id": 0,
  "paused_by_panel": true,            // false if the game was already paused — tells UI whether resuming is "its" concern
  "step": 143,
  "architecture": { /* same shape as the /architecture route */ },
  "activations": { /* see layers schema below */ }
}
```

Client → server to re-capture activations without closing the panel (game is already paused, so this just re-hooks forward() on the current tick):
```jsonc
{ "type": "refresh_activations", "agent_id": 0 }
```

Server → client (activation-only response):
```jsonc
{
  "type": "activations",
  "agent_id": 0,
  "step": 143,
  "layers": {
    "agent_encoder.0": { "shape": [8], "values": [ … ] },            // focal branch
    "agent_encoder.0/others": { "shape": [N, 8], "values": [[…]] },  // others branch (same module, different input)
    "attention_query": { "shape": [8], "values": [ … ] },
    "attention": { "shape": [8], "values": [ … ], "weights": { "shape": [N], "values": [ … ] } },
    "decision_net.0": { "shape": [16], "values": [ … ] },
    "decision_net.2": { "shape": [1], "values": [ … ] }
  }
}
```

Batch dim is squeezed out (always 1 for panel snapshots). Closing the panel does NOT auto-resume — the user clicks Play if they want to continue.

---

## Future-proofing — why this doesn't break when the architecture changes

The 95% path is pure introspection:
- `named_modules()` enumerates every `nn.Module` subclass, including user-defined ones.
- `named_parameters(recurse=False)` gives weight/bias shape and count.
- `named_children()` gives the parent/child hierarchy for collapse/expand.
- For `Linear`, `Conv1d/2d/3d`, `LayerNorm`, `Embedding`, `MultiheadAttention` etc. there's a small introspector registry (`LAYER_INTROSPECTORS: dict[type, Callable]`) that pulls `in_features`, `out_features`, `kernel_size`, `num_heads` etc. Unknown layer types fall back to a generic view that still shows type name + param count + weight shapes. **This is the only code that needs a one-line addition when a new layer type is introduced**, and it's additive — unknown types always render correctly, just without the fancy shape labels.

The 5% path — data-flow edges and inline tensor ops (the attention `bmm`/`softmax`/`bmm` block in `AgentPredictor.forward`) — cannot be recovered from `nn.Module` alone (they live in `forward` source code). Options:

- **A (recommended):** a tiny `ARCHITECTURE_TEMPLATES: dict[str, dict]` that maps `type(model).__name__` to a hand-written edge list + synthetic op nodes. Adds two keys when a new model type ships: `nodes` and `edges`. The **modules** inside each template are still introspected; only the wiring is declarative.
- **B (over-engineered):** parse `forward`'s AST, extract tensor ops. High effort, fragile. Skip.
- **C (fallback):** render modules side-by-side with no inter-module edges. Ugly but always works for unknown model types.

Ship with A + C: templated when known, side-by-side fallback otherwise. That's the full "auto-adapt" behavior.

---

## Phase 1 — Backend introspection (no UI yet)

### Step 1.1 — Install prerequisites

No backend deps. Skip.

### Step 1.2 — Create `tr_lbf_addon/nn_introspection.py`

Pure functions, no FastAPI imports, fully unit-testable.

Public surface:
```python
def build_architecture(model: nn.Module) -> dict:
    """Walk the module tree and produce the serializable architecture dict.
    Returns nodes + edges. Uses LAYER_INTROSPECTORS for per-type shape details
    and ARCHITECTURE_TEMPLATES for inter-module wiring."""

def get_weights(model: nn.Module, node_id: str) -> dict:
    """Return {weight: {shape, values, stats}, bias: {...}} for a leaf node.
    Raises KeyError if node_id isn't a Module path or has no parameters."""

def capture_activations(
    model: nn.Module,
    fruit_level: torch.Tensor,
    focal_features: torch.Tensor,
    others_features: torch.Tensor,
) -> dict:
    """Run one forward pass with forward hooks on every named_module;
    return {module_path: {shape, values}}. Hooks are always removed in a finally block."""

def architecture_hash(model: nn.Module) -> str:
    """Stable SHA1 over (module paths, types, param shapes). Changes iff the
    architecture changes — used by the frontend to invalidate cached graphs."""
```

Constants:
```python
LAYER_INTROSPECTORS: dict[type, Callable[[nn.Module], dict]] = {
    nn.Linear:    lambda m: {"in_features": m.in_features, "out_features": m.out_features},
    nn.ReLU:      lambda m: {},
    nn.Sigmoid:   lambda m: {},
    nn.Softmax:   lambda m: {"dim": m.dim},
    nn.Tanh:      lambda m: {},
    nn.LayerNorm: lambda m: {"normalized_shape": list(m.normalized_shape)},
    # Extend here when a new leaf layer type is introduced.
}

ARCHITECTURE_TEMPLATES: dict[str, dict] = {
    "AgentPredictor": {
        "inputs":  [ ... ],   # see data contract above
        "outputs": [ ... ],
        "synthetic_nodes": [
            { "id": "attention", "kind": "op", "type": "ScaledDotProduct", "param_count": 0 }
        ],
        "edges":   [ ... ],
    },
    # New top-level model types get an entry here. Until then, fallback C (no edges) kicks in.
}
```

Implementation notes:
- `build_architecture` composes introspected module nodes with the template's synthetic nodes + edges; if no template exists, nodes are returned flat and edges are omitted (frontend renders them in a column).
- `get_weights` resolves `node_id` by `model.get_submodule(node_id)`; formats values as nested Python lists (ok for these small dims — add downsampling later if dims grow past ~10k elements).
- `capture_activations` registers one hook per `named_module`; for `AgentPredictor` specifically, the hook on `agent_encoder` fires twice (focal, then others) — the second call key is suffixed `/others`.
- `architecture_hash`: hash over `[(path, type_name, tuple(p.shape for p in module.named_parameters(recurse=False)))]`. Parameter *values* do NOT affect the hash; only structure. Training-driven weight changes don't churn the UI.

### Step 1.3 — Tests `tr_lbf_addon/tests/test_nn_introspection.py`

Use the existing `AgentPredictor` as the primary fixture. Also test with a throw-away custom `nn.Module` containing an **unknown** layer type (e.g., a user-defined `nn.Module` subclass) to lock in the fallback behavior.

Test cases (class `TestBuildArchitecture`):
1. `test_agent_predictor_has_expected_top_level_modules` — asserts `agent_encoder`, `attention_query`, `decision_net` nodes exist.
2. `test_linear_layers_report_in_out_features` — check `in_features`/`out_features` for the three `Linear`s.
3. `test_param_count_matches_sum_of_weight_and_bias` — for each layer with params.
4. `test_template_edges_include_attention_branch` — the focal/others branch split from `agent_encoder`.
5. `test_unknown_model_type_falls_back_to_flat_nodes` — custom class → nodes returned, `edges == []`.
6. `test_unknown_layer_type_falls_back_to_generic_render` — custom `nn.Module` child → type name + param count only.
7. `test_hash_stable_across_calls` — two calls → same hash.
8. `test_hash_changes_when_embedding_dim_changes` — build two `AgentPredictor(embedding_dim=8)` and `AgentPredictor(embedding_dim=16)` → different hashes.
9. `test_hash_unchanged_after_training_step` — run one optimizer step → hash unchanged.

Class `TestGetWeights`:
10. `test_linear_weight_shape_matches_module` — `weight.shape == [out_features, in_features]`.
11. `test_raises_for_nonexistent_node` — `get_weights(model, "nope.nope")` raises.
12. `test_raises_for_layer_without_parameters` — `get_weights(model, "decision_net.1")` (ReLU) raises.
13. `test_stats_computed_correctly` — min/max/mean on a known-init tensor.

Class `TestCaptureActivations`:
14. `test_returns_entry_for_every_named_module` — keys ⊇ `{agent_encoder.0, attention_query, decision_net.0, decision_net.2}`.
15. `test_shapes_match_forward_trace` — `decision_net.2` shape is `[1]` after squeezing batch.
16. `test_hooks_removed_after_call` — `len(list(model._forward_hooks.items())) == 0` after call (iterate modules).
17. `test_hooks_removed_on_exception` — pass mismatched input shapes, catch `RuntimeError`, assert hooks cleaned up.
18. `test_agent_encoder_fires_twice_with_focal_and_others_keys` — for `AgentPredictor` only.

### Step 1.4 — Verify

```bash
python -m pytest tr_lbf_addon/tests/test_nn_introspection.py -q
```

All ~18 tests green.

---

## Phase 2 — Backend HTTP + WebSocket wiring

### Step 2.1 — `web/backend/nn_routes.py`

Exports an `APIRouter` and a `handle_ws_message(runner, msg, send_fn)` coroutine. No logic duplicated from `nn_introspection`.

Routes:
- `GET /api/nn/{agent_id}/architecture` → calls `build_architecture(runner.lbf_gym.agents[agent_id].neural_network)`; 404 if agent out of range or has no NN; 200 with the dict.
- `GET /api/nn/{agent_id}/weights/{node_id:path}` → calls `get_weights(...)`. `node_id` is `:path` to allow dots.

WebSocket message types (to be added to the handler loop in `server.py`):
- `open_nn_panel` — force-pauses the game loop (same code path as the existing `pause` message: set `paused = True`, `stop_loop()`), then builds the inputs from `runner.lbf_gym` current state (reuse `_build_nn_input` from `lbf_elements.py` for consistency), runs `capture_activations`, and sends one combined `{"type": "nn_panel_ready", ...}` payload carrying architecture + activations. The `paused_by_panel` field distinguishes panel-triggered pause from user-triggered pause.
- `refresh_activations` — game is already paused; re-runs the hooked forward pass on the current tick and sends `{"type": "activations", ...}`.

There is intentionally **no streaming** message type. The frontend never asks for live activations.

### Step 2.2 — Modify `web/backend/server.py`

Add:
```python
from web.backend.nn_routes import router as nn_router, handle_nn_ws
app.include_router(nn_router)
```

Inside the WebSocket message loop, add message-type dispatch for `open_nn_panel` and `refresh_activations` (delegate to `handle_nn_ws`, passing in the closure that can mutate `paused`/`loop_task` — treat it the same way the existing `pause` handler does).

### Step 2.3 — Modify `web/backend/serializer.py`

Add to each agent's frame dict:
```python
"architecture_hash": architecture_hash(agent.neural_network) if agent.neural_network else None,
```

### Step 2.4 — Backend tests `web/backend/tests/test_nn_routes.py`

First tests in the web backend — create `web/backend/tests/__init__.py` and `conftest.py` as needed.

Use FastAPI's `TestClient`. Build a `GameRunner` with `default_params()`, call `reset()` to materialize agents, then hit the routes.

- `test_get_architecture_for_agent_0_has_nodes` — non-empty `nodes`.
- `test_get_architecture_404_for_out_of_range_agent`.
- `test_get_weights_linear_layer_returns_shape_and_values`.
- `test_get_weights_404_for_activation_layer`.
- `test_architecture_hash_on_frame_matches_route` — WebSocket frame's `architecture_hash` == route's.

### Step 2.5 — Verify

```bash
python -m pytest tr_lbf_addon/tests/test_nn_introspection.py web/backend/tests/ -q
python -m uvicorn web.backend.server:app --port 8000   # smoke: GET /api/nn/0/architecture in browser
```

---

## Phase 3 — Frontend: toolbar toggle + architecture panel skeleton

### Step 3.1 — Add `@xyflow/react`

```bash
cd web/frontend && npm install @xyflow/react
```

(v12+. v12 has native support for sub-flows / grouped nodes for expand/collapse.)

### Step 3.2 — Types `web/frontend/src/types/nn.ts`

Mirror the backend contract as TypeScript interfaces. Source of truth is the backend; if backend changes, update here in the same commit.

### Step 3.3 — Toolbar toggle

Modify `web/frontend/src/components/Toolbar.tsx`: add a **NN** button next to the Settings/Metrics toggles (same styling). Props gain `nnOpen: boolean`, `onToggleNN: () => void`.

### Step 3.4 — Panel shell and open-handshake

`NNVizPanel.tsx`: overlay in the same style as `SettingsPanel.tsx`. Contains:
- Header: agent selector (`<select>` populated from `frame.agents`), **Refresh snapshot** button (always enabled while panel is open — the panel guarantees paused state), close button.
- Body: left ~70% `ArchitectureGraph`, right ~30% `DetailPanel`.

Open flow:
1. Toolbar **NN** toggle flips `nnOpen` to `true` in `App.tsx` state.
2. On mount, `NNVizPanel` sends `{ type: "open_nn_panel", agent_id: selectedAgentId }` over the WebSocket.
3. The backend pauses the game loop and responds with `{ type: "nn_panel_ready", architecture, activations, paused_by_panel }`.
4. Panel renders the architecture graph + feeds activations into `DetailPanel`.

Agent switching while the panel is open: selecting a new `agent_id` from the header dropdown sends another `open_nn_panel` for that agent. Server is idempotent (already paused) and returns the fresh payload.

Close flow: just unmounts the panel. Game remains paused. A one-line hint in the header reads "Game paused while viewer is open — close and press Play to resume." if `paused_by_panel` was true, hidden otherwise.

`App.tsx`: add `nnOpen` and `selectedAgentId` state, wire toggle, render `<NNVizPanel>` when open.

### Step 3.5 — `useNNArchitecture.ts` hook

Triggers the `open_nn_panel` WebSocket handshake on mount (and on `agentId` change); caches the resulting architecture keyed by `agentId + architectureHash`. `architectureHash` comes from the frame (`frame.agents[i].architecture_hash`) so stale caches are invalidated automatically after evolution between episodes.

### Step 3.6 — Verify

Run `start.bat`. Play for a few ticks, then toggle the NN panel. Expect:
- Game pauses (Play/Pause button reflects the state).
- Panel shows the JSON architecture raw (e.g., in a `<pre>`) for now — no graph yet.
- Close panel → game stays paused.
- Smoke check: restart with a mutated saved genome → reopening shows the new architecture hash.

---

## Phase 4 — The graph itself (React Flow)

### Step 4.1 — `ArchitectureGraph.tsx`

- Convert `nodes`/`edges` from the backend schema to React Flow's format.
- Node layout: **dagre** (left-to-right, LR). Install `dagre` if needed. Re-layout when architecture changes.
- Parent/child: nodes with `parent` ≠ `null` become React Flow **group** children. Clicking a group collapses/expands via local state (hide child nodes + their edges; replace incoming/outgoing edges with edges to/from the group node).
- Pan + zoom: React Flow built-ins. Initial zoom fits to screen.
- Node type dispatch: `ModuleNode` for `kind: "module"`, `LayerNode` for `kind: "layer"`, `IONode` for `kind: "input" | "output"`, a fourth `OpNode` for `kind: "op"`.

### Step 4.2 — Node components

`ModuleNode` — colored by `type` (Sequential, ModuleList, AgentPredictor, custom…). Shows name, type, param count, an expand/collapse chevron.

`LayerNode` — colored by type family (linear/activation/norm/embedding). Shows name, type, in→out shape, param count. On hover, tooltip with more detail. On click, sets selected → `DetailPanel` populates.

`IONode` — small pill with shape label.

`OpNode` — dashed border (denotes "not an nn.Module, derived"). Note tooltip.

### Step 4.3 — Verify

Run app. Open panel. Expect the `AgentPredictor` graph with three top-level groups (`agent_encoder`, `attention_query`, `decision_net`) plus I/O nodes. Click `agent_encoder` → its `Linear` and `ReLU` expand. Pan/zoom work smoothly.

---

## Phase 5 — Detail panel: weights + activations

### Step 5.1 — `DetailPanel.tsx`

Right-hand pane. Driven by the `selectedNode` in `ArchitectureGraph`. Layouts:

- **Module selected:** summary (type, param count, child list); no heatmap.
- **Linear layer selected:** `WeightHeatmap` for weight (shape `[out, in]`), `ActivationBars` for bias (1D), stats row. If activations captured: `ActivationBars` for the layer's output activation.
- **Activation/op layer selected:** just the activation values if captured, otherwise "no parameters" placeholder.
- **I/O node selected:** shape + semantic description.

### Step 5.2 — `WeightHeatmap.tsx` (Canvas)

- Draws weight matrix as a heatmap using a diverging colormap centered at 0 (blue negative, red positive, white 0).
- Uses the existing canvas convention (follows `renderer.ts`): takes `ctx`, computes a `(out, in)` grid, draws filled rectangles per cell.
- Mouse hover → shows `(i, j): 0.123` tooltip.
- Responsive: resizes on panel resize.

### Step 5.3 — `ActivationBars.tsx` (Recharts)

- BarChart of values; X = index, Y = value. Min/Max/mean as a horizontal dashed line.
- Used for: bias vectors, output activations of 1D layers, attention weights.

### Step 5.4 — `useNNWeights.ts` hook

Lazy fetch of `GET /api/nn/{agentId}/weights/{nodeId}` when `DetailPanel` gets a new `selectedNode` with `has_weight === true`. Cache keyed by `(agentId, architectureHash, nodeId)`.

### Step 5.5 — Activation refresh flow

- Activations arrive with the `nn_panel_ready` payload on panel open (no separate user action).
- **Refresh snapshot** button in the panel header → sends `{ type: "refresh_activations", agent_id }` → incoming `{ type: "activations", ... }` message → replace the cached activations in panel state. Since the game is always paused while the panel is open, "stale activations" is not a concept — every snapshot is the current frozen state.
- If the agent selector changes: the new `open_nn_panel` handshake brings a fresh snapshot for the new agent.

### Step 5.6 — Verify

1. Load app. Let it play for a few ticks (so the NN has warmed up). Open NN panel → game auto-pauses; architecture graph renders; activations are populated on open.
2. Click `agent_encoder.0` → weight heatmap renders, bias bar chart renders, focal-branch activations appear below.
3. Close panel → game stays paused. Press Play → game resumes.
4. Pause manually, re-open panel → `paused_by_panel` is false (user had paused first); close hint suppresses the "resume" copy.
5. Switch agent in dropdown → graph + activations refresh for the new agent.
6. After a generation change (between episodes): re-opening the panel on a child agent shows the new `architecture_hash` and updated graph.

---

## Phase 6 — Polish

### Step 6.1 — Agent selector shows architecture diff highlight

When the selected agent's `architecture_hash` differs from any other agent's, the selector lists "Agent 0 (dim=8)", "Agent 1 (dim=16)" so the user can pick the divergent one.

### Step 6.2 — Empty / error states

- Agent has no NN yet (`neural_network is None`): panel shows "Network not initialized — play a few steps to warm up."
- Agent is dead: selector greys the option; if already selected, panel shows "Agent dead — select another from the dropdown."
- Backend 5xx during handshake: inline error with a Retry button. The retry re-sends `open_nn_panel`.

### Step 6.3 — README

Append to `README.md`:

```markdown
## Neural Network Visualization

Toggle the **NN** button in the toolbar to inspect any agent's `AgentPredictor`. Opening the panel auto-pauses the game and freezes the current tick; closing the panel leaves the game paused so you can continue with Play.

- Zoom through the module graph (system → modules → layers) with mouse/trackpad.
- Click a layer to see its weight heatmap and bias chart alongside the captured activations from the frozen tick.
- Use **Refresh snapshot** in the panel header to re-run a forward pass on the same tick (useful after a manual Step).
- Switch between agents from the dropdown — each agent has its own evolved architecture.

The graph is built by live `torch.nn.Module` introspection, so new layer types and dim changes show up automatically. Wiring between modules (e.g. the attention branch in `AgentPredictor`) uses a small declarative template in `tr_lbf_addon/nn_introspection.py::ARCHITECTURE_TEMPLATES` — the same dict doubles as the canonical text-based spec of the network topology.
```

### Step 6.4 — memory/project_state.md

Add a block summarizing the viz (routes, components, hooks, dep bump) and bump test counts.

---

## Verification (end-to-end)

1. **Unit tests:** `python -m pytest tr_lbf_addon/tests/ web/backend/tests/ -q` — all green.
2. **Type check:** `cd web/frontend && npm run build` — compiles without TS errors.
3. **Manual smoke test (golden path):**
   - `start.bat`
   - NN toggle → panel opens with agent 0 graph.
   - Expand each module group → leaf layers visible.
   - Click `decision_net.0` → heatmap (16×11) renders without blanks.
   - Open panel → game auto-pauses; activation bars populate immediately (no manual capture needed).
   - Close panel → game stays paused; Play resumes it.
   - Run until reproduce() triggers, re-open panel on a child agent → new architecture hash, graph re-renders.
4. **Manual edge cases:**
   - Resize window → panel + graph both re-layout.
   - Agent dies mid-viz → selector greys the dead agent; if already selected, panel shows "agent dead" state.
   - Load a saved genome with a different `embedding_dim` → architecture + weights update on next panel open.

---

## Critical files quick-reference

- Source of truth for model: [tr_lbf_addon/neuroevolution.py](tr_lbf_addon/neuroevolution.py) — `AgentPredictor` at lines 18-110.
- NN instantiation per agent: [tr_lbf_addon/lbf_elements.py](tr_lbf_addon/lbf_elements.py) — `init_neural_network` at lines 635-650.
- NN input builder (reused by `capture_activations`): [tr_lbf_addon/lbf_elements.py](tr_lbf_addon/lbf_elements.py) — `_build_nn_input` at lines 28-87.
- Genome save/load (architecture hash must stay consistent with save format): [tr_lbf_addon/game_runner.py](tr_lbf_addon/game_runner.py) — lines 412-459.
- WebSocket + serializer entry points: [web/backend/server.py](web/backend/server.py), [web/backend/serializer.py](web/backend/serializer.py).
- Frontend rendering conventions: [web/frontend/src/lib/renderer.ts](web/frontend/src/lib/renderer.ts), [web/frontend/src/components/SettingsPanel.tsx](web/frontend/src/components/SettingsPanel.tsx).

## Risks and mitigations

- **React Flow bundle size (~100kb gzip).** Acceptable for a dev/analysis tool; guarded by the NN toggle so it's not loaded until opened (code-split with `React.lazy` if the budget is tight — not required for v1).
- **Very wide dims (future evolution):** heatmap becomes unreadable past ~10k cells. Mitigation: downsample the weight tensor server-side when `shape > threshold` and flag `downsampled: true`; add tiled rendering in a later phase.
- **Forward-pass cost on capture:** negligible for these dims; becomes relevant only if dims grow 10×.
- **`ARCHITECTURE_TEMPLATES` drift:** if someone changes `AgentPredictor.forward` without updating the template, edges will be wrong but nodes still correct. Mitigation: a template-drift test that runs the model, compares captured activation keys to template edges, fails loudly if they disagree.

## Deliberately NOT doing

- No on-graph architecture editing.
- No training-curve visualization (`MetricsPanel` already covers learning life; the user explicitly prefers keeping these two concerns separate).
- **No live streaming of any NN data.** Opening the panel pauses the game and freezes a snapshot; all inspection is on that frozen state. Closing the panel leaves the game paused for the user to resume manually.
- No comparison view across agents in v1 (possible follow-up; needs a diff layout).
