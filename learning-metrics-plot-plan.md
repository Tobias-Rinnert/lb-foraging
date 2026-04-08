# Animated Learning Metrics Plot — Implementation Plan

## Context

The LB Foraging web app currently shows only the current game state. The user wants a visual signal that their agents are actually learning — "an animated plot of learning rate" — and they explicitly flagged that **this is the first of many future analysis plots, so the foundation must be extensible**.

Today there is no metric tracking: `game_runner.py` overwrites `self.rewards` every step, `Agent.learn()` computes MSE loss and then discards it (`lbf_elements.py:443`), and the frontend `useGameSocket` keeps only the latest frame. We need to introduce a proper metrics pipeline end-to-end — backend tracker → WebSocket protocol → frontend history hook → chart components — structured so that adding "plot N+1" in the future means writing one new chart component and registering one new series, not re-architecting anything.

## Decisions (confirmed with user)

- **Metrics tracked (v1):** mean episode return, total episode return, mean NN loss, per-agent NN loss.
- **Chart library:** Recharts (~95 KB, React 19 compatible, declarative).
- **Layout:** collapsible bottom drawer toggled from the Toolbar.
- **Reset policy:** metric history persists across episode boundaries; cleared only on `apply_params` / `rebuild`.
- **X-axis:** episode index (clean monotonic learning curve). Step-indexed series can be added later.

## Architecture Overview

```
          ┌──────────────────────────────────────────────────────────┐
 backend: │  Agent.learn() → list[float]  (new return value)         │
          │            │                                              │
          │            ▼                                              │
          │  LBF_GYM.update_observation() accumulates into            │
          │    self.last_step_losses_per_agent: list[list[float]]     │
          │            │                                              │
          │            ▼                                              │
          │  GameRunner.step()                                        │
          │    • accumulates self._cumulative_rewards                 │
          │    • pulls last_step_losses from lbf_gym                  │
          │    • calls self.metrics.record_step_losses(...)           │
          │    • on episode_over:                                     │
          │        self.metrics.record_episode_end(...)               │
          │            │                                              │
          │            ▼                                              │
          │  MetricsTracker (persists across episodes)                │
          │    - episode_index                                        │
          │    - series: dict[str, MetricSeries]                      │
          │    - _pending_episode_losses (accumulator)                │
          │            │                                              │
          │            ▼                                              │
          │  metrics_serializer.py                                    │
          │    - serialize_metrics_snapshot(tracker)                  │
          │    - serialize_metrics_latest(tracker, new_episode)       │
          │            │                                              │
          └────────────┼──────────────────────────────────────────────┘
                       ▼
                 WebSocket JSON
                       │
          ┌────────────┼──────────────────────────────────────────────┐
 frontend:│  useGameSocket → exposes {frame, lastMessage, connected}  │
          │            │                                              │
          │            ▼                                              │
          │  useMetricsHistory(lastMessage)                           │
          │    handles: metrics_snapshot / metrics_cleared /          │
          │             frame.metrics_latest                          │
          │    returns: MetricsState { episodeIndex, series }         │
          │            │                                              │
          │            ▼                                              │
          │  MetricsPanel (bottom drawer)                             │
          │     └─ LearningChart                                      │
          │           └─ LineChart (generic Recharts wrapper)         │
          └──────────────────────────────────────────────────────────┘
```

## File-by-file changes

All paths relative to `C:\Users\bobis\development\lb-foraging\`.

### Backend

1. **CREATE `tr_lbf_addon/metrics_tracker.py`** (~150 LOC)
   - `MetricSeries` dataclass: `name: str`, `unit: str`, `max_points: int = 2000`, `points: list[tuple[float, float]]`. Methods: `append(x, y)` with rolling-window slice, `clear()`, `to_dict()`.
   - `MetricsTracker`:
     - `episode_index: int = 0`
     - `series: dict[str, MetricSeries]` initialised with four keys: `episode_return_mean`, `episode_return_total`, `nn_loss_mean`, `nn_loss_agent_{id}` — the per-agent series are created lazily on first `record_step_losses` call so the tracker doesn't need to know player count upfront.
     - `_pending_losses_per_agent: dict[int, list[float]]` — accumulates losses within an episode.
     - `record_step_losses(losses_per_agent: dict[int, list[float]]) -> None` — extends the pending buffer.
     - `record_episode_end(cumulative_rewards: list[float]) -> None` — computes mean & total returns, computes mean loss overall and per-agent, appends one point per series at `x = episode_index`, increments `episode_index`, clears `_pending_losses_per_agent`.
     - `snapshot() -> dict` — returns `{episode_index, series: {key: series.to_dict()}}`.
     - `latest_values() -> dict | None` — returns the last appended point per series, or `None` if no episode completed since last call. Uses an internal `_dirty: bool` flag set by `record_episode_end` and cleared when `latest_values()` is called.
     - `clear() -> None` — full reset.
   - **No while loops** (rolling window via `list.__getitem__` slice). Loops over agents are for-loops.

2. **MODIFY `tr_lbf_addon/lbf_elements.py`** — `Agent.learn()` at lines 427-446.
   - Change signature to return `list[float]`.
   - Append `float(loss.detach().item())` to a local `losses` list each iteration.
   - Return `losses` (empty list if `neural_network is None` or no labeled predictions).
   - No other behavioural change.

3. **MODIFY `tr_lbf_addon/lbf_gym.py`** — `update_observation()` around line 117-135.
   - Add `self.last_step_losses_per_agent: dict[int, list[float]] = {}` initialisation in `__init__` (or set at the start of `update_observation`).
   - At line 135: replace `agent.learn()` with `self.last_step_losses_per_agent[agent.id] = agent.learn()`.
   - No other behavioural change.

4. **MODIFY `tr_lbf_addon/game_runner.py`**
   - Import `MetricsTracker`.
   - `__init__`: add `self.metrics = MetricsTracker()` and `self._cumulative_rewards: list[float] = []`.
   - `reset()`: zero `self._cumulative_rewards` (length = `number_players`). Do NOT clear metrics.
   - `step()`:
     - After `env.step`, for-loop over per-agent rewards to accumulate into `self._cumulative_rewards` (vectorisable with `numpy`; use list comprehension for zero-dep simplicity).
     - After the env step, read `self.lbf_gym.last_step_losses_per_agent` and call `self.metrics.record_step_losses(...)`.
     - If `self.episode_over`, call `self.metrics.record_episode_end(self._cumulative_rewards)`.
   - `rebuild()`: call `self.metrics.clear()` BEFORE `self._build_env()`.

5. **CREATE `web/backend/metrics_serializer.py`** (~40 LOC)
   - `serialize_metrics_snapshot(tracker: MetricsTracker) -> dict` → `{type: "metrics_snapshot", episode_index, series}`.
   - `serialize_metrics_latest(tracker: MetricsTracker) -> dict | None` → returns latest-values dict if tracker is dirty, else `None`.

6. **MODIFY `web/backend/serializer.py`**
   - In `serialize_frame`, after building the frame dict, call `serialize_metrics_latest(runner.metrics)` and, if non-None, embed as `frame["metrics_latest"]`. Keeps existing callers back-compatible (field is optional).

7. **MODIFY `web/backend/server.py`**
   - Import `serialize_metrics_snapshot`.
   - After initial `runner.reset()` + `send_frame()`, also send `serialize_metrics_snapshot(runner.metrics)`.
   - In `apply_params` handler: after `runner.rebuild(...)`, send `{type: "metrics_cleared"}`, then `serialize_metrics_snapshot(runner.metrics)`, then the normal `send_frame()`.
   - Handle new client message `request_metrics_snapshot` → send snapshot (for reconnect robustness).

### Tests

8. **CREATE `tr_lbf_addon/tests/test_metrics_tracker.py`** (~120 LOC)
   - `test_metric_series_append`
   - `test_metric_series_rolling_window_drops_oldest` (max_points=5, append 10, assert length and x of first/last)
   - `test_tracker_record_episode_end_advances_index`
   - `test_tracker_record_episode_end_computes_mean_and_total_return`
   - `test_tracker_loss_aggregation_mean` (input: `{0: [0.1, 0.2], 1: [0.3]}` → overall mean 0.2, per-agent [0.15, 0.3])
   - `test_tracker_clear_resets_episode_and_series`
   - `test_tracker_latest_values_dirty_flag` (latest returns dict once, then None)
   - `test_tracker_snapshot_round_trip_json_serializable` (json.dumps succeeds)
   - `test_tracker_per_agent_series_created_lazily`

9. **CREATE `tr_lbf_addon/tests/test_game_runner_metrics.py`** (~80 LOC)
   - `test_game_runner_accumulates_cumulative_rewards_across_steps` (mock env)
   - `test_game_runner_records_episode_end_on_episode_over`
   - `test_game_runner_rebuild_clears_metrics`
   - `test_game_runner_captures_nn_losses_from_lbf_gym`

### Frontend

10. **CREATE `web/frontend/src/types/metrics.ts`**
    ```ts
    export interface MetricPoint { x: number; y: number; }
    export interface MetricSeriesData {
      key: string;
      name: string;
      unit: string;
      color: string;
      points: MetricPoint[];
    }
    export interface MetricsState {
      episodeIndex: number;
      series: Record<string, MetricSeriesData>;
    }
    ```

11. **CREATE `web/frontend/src/constants/agentColors.ts`**
    - Mirror of `_AGENT_COLOURS` from `serializer.py`. Comment points at the Python source as the canonical definition.
    - Export `getAgentColor(id: number): string`.

12. **MODIFY `web/frontend/src/types/game.ts`**
    - Add optional field on `GameFrame`: `metrics_latest?: { episode_index: number; values: Record<string, number>; per_agent?: Record<string, Record<number, number>> }`.

13. **MODIFY `web/frontend/src/hooks/useGameSocket.ts`**
    - Add `lastMessage: unknown` to the returned object (the latest parsed JSON regardless of type).
    - Keep existing `frame` field (only updated on game-frame messages) for back-compat.

14. **CREATE `web/frontend/src/hooks/useMetricsHistory.ts`** (~90 LOC)
    - Signature: `useMetricsHistory(lastMessage: unknown, maxPointsPerSeries = 2000): MetricsState`.
    - Uses `useState<MetricsState>({ episodeIndex: 0, series: {} })` and `useEffect` on `lastMessage`.
    - Message dispatch:
      - `type === "metrics_snapshot"` → replace state with snapshot, mapping each series to `MetricSeriesData` with a colour assigned from a palette (see below).
      - `type === "metrics_cleared"` → reset state.
      - object has `metrics_latest` → append one point per series key, rolling window via `slice(-maxPointsPerSeries)`.
    - Colour assignment: `episode_return_mean` → `--accent`, `episode_return_total` → `#457B9D`, `nn_loss_mean` → `#2A9D8F`, `nn_loss_agent_{id}` → `getAgentColor(id)`.

15. **Install Recharts**
    - `cd web/frontend && npm install recharts`
    - Verify React 19 peer-dep satisfied (Recharts 2.13+).

16. **CREATE `web/frontend/src/components/charts/LineChart.tsx`** (~70 LOC)
    - Generic wrapper around Recharts' `<ResponsiveContainer><LineChart>`.
    - Props: `{ series: MetricSeriesData[]; xLabel?: string; yLabel?: string; height?: number; animate?: boolean; yScale?: "linear" | "log" }`.
    - Dark-theme styling via CSS vars (`var(--bg-surface)`, `var(--text-primary)`, `var(--border)`).
    - One `<Line>` per series with `stroke={series.color}`, `isAnimationActive={animate}`, `dot={false}`.
    - Tooltip enabled, legend at top.

17. **CREATE `web/frontend/src/components/charts/LearningChart.tsx`** (~90 LOC)
    - Props: `{ metrics: MetricsState }` (pure, no own subscription).
    - Internal state: which series keys are visible (checkboxes above the chart). Defaults visible: `episode_return_mean`, `nn_loss_mean`.
    - Derives `selectedSeries: MetricSeriesData[]` from `metrics.series` filtered by visible keys.
    - Renders `<LineChart series={selectedSeries} xLabel="Episode" yLabel="Value" />`.
    - Log-scale toggle for the loss curve (optional, cheap to add).

18. **CREATE `web/frontend/src/components/charts/MetricsPanel.tsx`** (~60 LOC)
    - The drawer container. Renders header (title + close button), body with one or more charts, footer placeholder.
    - Props: `{ metrics: MetricsState; onClose: () => void }`.
    - V1 body: single `<LearningChart metrics={metrics} />`. Adding plot #2 later = drop another `<XxxChart>` below it or introduce tabs.

19. **MODIFY `web/frontend/src/components/Toolbar.tsx`**
    - Add a "Metrics" toggle button (icon: small bar-chart SVG or just text "Metrics"). Props extended with `metricsOpen: bool` and `onToggleMetrics: () => void`.

20. **MODIFY `web/frontend/src/App.tsx`**
    - `const [metricsOpen, setMetricsOpen] = useState(false);`
    - Call `useMetricsHistory(lastMessage)` (from extended `useGameSocket`).
    - Pass `metricsOpen` / `onToggleMetrics` into `<Toolbar>`.
    - Render `<MetricsPanel metrics={metrics} onClose={() => setMetricsOpen(false)} />` inside a new `<div className={`metrics-drawer ${metricsOpen ? "" : "collapsed"}`}>` slot between `.main-area` and `<StatusBar>`.

21. **MODIFY `web/frontend/src/App.css`**
    ```css
    .metrics-drawer {
      height: 320px;
      border-top: 1px solid var(--border);
      background: var(--bg-secondary);
      overflow: hidden;
      transition: height 0.2s ease;
      flex-shrink: 0;
    }
    .metrics-drawer.collapsed { height: 0; }
    .metrics-panel { display: flex; flex-direction: column; height: 100%; padding: 12px 16px; }
    .metrics-panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .chart-container { flex: 1; min-height: 0; }
    .chart-toggles { display: flex; gap: 12px; margin-bottom: 8px; font-size: 12px; }
    ```

### Documentation

22. **MODIFY `README.md`**
    - Add a "Metrics & Analysis Plots" section under "Running the Web App" describing the metrics drawer, the four v1 metrics, and a short "How to add a new plot" developer note pointing at `MetricsTracker.series`, `useMetricsHistory`, and `charts/`.
    - Remove the "add animated plot of learning rate to front end" TODO.

## Critical code paths referenced

- `C:\Users\bobis\development\lb-foraging\tr_lbf_addon\game_runner.py` lines 49-75 — step/reset/rebuild hooks.
- `C:\Users\bobis\development\lb-foraging\tr_lbf_addon\lbf_elements.py` lines 427-446 — `Agent.learn()` loss capture.
- `C:\Users\bobis\development\lb-foraging\tr_lbf_addon\lbf_gym.py` line 135 — `agent.learn()` call site.
- `C:\Users\bobis\development\lb-foraging\web\backend\serializer.py` lines 23-62 — existing frame shape.
- `C:\Users\bobis\development\lb-foraging\web\backend\server.py` lines 37-122 — message handlers, connect lifecycle.
- `C:\Users\bobis\development\lb-foraging\web\frontend\src\App.tsx` — layout root.
- `C:\Users\bobis\development\lb-foraging\web\frontend\src\hooks\useGameSocket.ts` — hook to extend.
- `C:\Users\bobis\development\lb-foraging\web\frontend\src\components\SettingsPanel.tsx` — panel pattern to mirror.

## Verification plan

### Automated
1. `pytest tr_lbf_addon/tests/test_metrics_tracker.py -v`
2. `pytest tr_lbf_addon/tests/test_game_runner_metrics.py -v`
3. `pytest tr_lbf_addon/tests/ -v` — confirm no regressions in existing tests.
4. `cd web/frontend && npm run lint` — confirm TypeScript / ESLint clean.
5. `cd web/frontend && npm run build` — confirm Vite build succeeds.

### Manual end-to-end
1. Run `start.bat`, wait for backend + frontend + browser to open.
2. Open Settings → reduce `max_episode_steps` to ~20 so episodes complete quickly → Apply.
3. Click the new **Metrics** toolbar button → drawer opens, chart empty with "Episode" axis.
4. Click Play. After each episode ends, confirm a new point appears on the `episode_return_mean` line and the `nn_loss_mean` line, and the chart animates in.
5. Run 10+ episodes. Sanity-check that return trends upward (or is at least non-zero) and loss trends downward — confirms the metrics pipeline is plumbed correctly, even if absolute learning is slow.
6. Toggle per-agent loss visibility on → confirm five lines appear (one per agent) with distinct colors matching the game board.
7. Click Apply Params (change any setting) → confirm chart clears immediately.
8. Hard-refresh the browser mid-run → confirm the drawer re-opens empty and starts rebuilding as new episodes complete (server runner is fresh per connection; this is expected v1 behaviour).
9. Confirm the drawer toggle animates smoothly and doesn't overlap the status bar or toolbar.

## Open design notes (non-blocking)

- **Per-connection runner** means metrics reset on browser reconnect. Fine for v1; later we could move the runner to a shared singleton or persist metrics to disk.
- **Metric computation extensibility:** adding a new metric later means (a) add one key to `MetricsTracker.series`, (b) call `record_*` from wherever the data is computed, (c) add one `<LineChart>` or checkbox to `LearningChart.tsx` — or add a whole new `<FooChart>` to `MetricsPanel.tsx`. No other file needs to change.
- **Log-scale toggle** on the loss curve is tiny to add and genuinely useful; included as an optional prop on `LineChart`.
- **Drawer resizing** skipped in v1 (fixed 320 px). If the user wants a drag handle, it's a small follow-up.
