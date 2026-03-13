# Design: LBF Game App

**Date:** 2026-03-13
**Status:** Approved

## Summary

A desktop Tkinter app that runs and visualises the LBF foraging game. Parameters are
configurable via a floating settings popup. The game board is drawn directly on a
Tkinter Canvas with zoom and pan support.

---

## Architecture (MVC, Option B)

Four files under `tr_lbf_addon/app/`:

| File | Role |
|------|------|
| `game_runner.py` | **Model** — wraps `LBF_GYM` + gym env. Holds all parameters, builds/rebuilds the env, steps the game, exposes current state. |
| `board_canvas.py` | **View** — `tk.Canvas` subclass. Stateless `draw(state)` call redraws grid, agents, fruits, target arrows. Handles zoom + pan via mouse wheel and drag. |
| `param_panel.py` | **View** — `tk.Toplevel` popup. Grouped `Spinbox`/`Scale`/`Checkbutton` widgets for all env parameters. Calls `on_apply(params)` callback on "Apply & Restart". |
| `app.py` | **Controller** — creates root window, wires the three pieces together, owns the `canvas.after()` game loop. Entry point: `python tr_lbf_addon/app/app.py`. |

---

## Window Layout

```
┌─────────────────────────────────────────────────────────┐
│  [▶ Play] [⏸ Pause] [→ Step]  Speed: ──●──  [⊞ Fit]  [⚙ Settings]  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                   GAME BOARD                            │
│               (Tkinter Canvas, 600×600px)               │
│                                                         │
│   Agents = coloured rounded squares + ID + level text   │
│   Fruits = orange circles + level text                  │
│   Free slots = small grey dots                          │
│   Target = dashed line from agent to fruit              │
│                                                         │
└─────────────────────────────────────────────────────────┘
│ Step 12/50  |  Agent 0: 0.4   Agent 1: 0.2  …          │
└─────────────────────────────────────────────────────────┘
```

### Settings Popup (Toplevel, non-modal)

Opens on **⚙ Settings**, stays open until user closes it.

```
┌─ Settings ──────────────────────────────┐
│  ▼ Environment                          │
│    Grid size:      [8 ▲▼]               │
│    Max steps:      [50 ▲▼]              │
│    Sight:          [0 ▲▼]               │
│    Coop mode:      [☑]                  │
│                                         │
│  ▼ Players                              │
│    Count:          [5 ▲▼]               │
│    Min level:      [1 ▲▼]               │
│    Max level:      [1 ▲▼]               │
│                                         │
│  ▼ Food                                 │
│    Max food:       [8 ▲▼]               │
│    Min level:      [1 ▲▼]               │
│    Max level:      [1 ▲▼]               │
│                                         │
│  ▼ Advanced                             │
│    Penalty:        ──●────  0.0         │
│    Normalize reward:  [☑]               │
│    Observe levels:    [☑]               │
│    Full info mode:    [☑]               │
│                                         │
│  [  Apply & Restart  ]   [Close]        │
└─────────────────────────────────────────┘
```

Each section is collapsible via the `▼` toggle. "Apply & Restart" calls
`runner.rebuild(params)` and resets the game loop.

---

## Board Canvas Rendering

**Canvas size:** fixed 600×600px — pixel count never changes, redraw is O(entities) not O(N²).

**Cell size:** `(600 / field_size) * zoom_level`

### Drawing layers (in order)
1. Grey grid lines
2. Orange fruit circles + level number
3. Small grey dots at free loading slots
4. Coloured rounded-rectangle agents + ID + level text (10-colour palette)
5. Dashed target arrows (agent → fruit)

### Detail thresholds (effective cell size after zoom)
| Effective cell px | What is shown |
|---|---|
| ≥ 40 | All labels (agent ID + level, fruit level) |
| ≥ 20 | Shapes + dots, no text |
| < 20 | Agents as dots, fruits as filled cells — no text |

Target arrows only drawn when effective cell ≥ 30px.

### Zoom + pan
- `zoom_level` (float, default `1.0`) — multiplies cell size
- **Mouse wheel** — zoom in/out centred on cursor
- **Click + drag** — pan when zoomed in
- **[⊞ Fit]** — resets zoom to `1.0` and recentres
- Items outside the visible canvas area are skipped before drawing (cheap bounds check)

---

## Game Loop

`GameRunner` owns the gym env + `LBF_GYM` instance. `app.py` drives:

```python
def step():
    if paused:
        return
    runner.step()                         # update_observation → choose_actions → env.step
    canvas.draw(runner.state)
    status_bar.update(runner.step_count, runner.rewards)
    if runner.episode_over:
        show_game_over_overlay()
    else:
        root.after(speed_ms, step)        # schedule next step
```

### Controls
| Control | Behaviour |
|---|---|
| ▶ Play | Starts `root.after()` loop; no-op if already running |
| ⏸ Pause | Sets `paused = True`; loop exits on next call |
| → Step | Calls `step()` once directly, ignores `paused` |
| Speed slider | Updates `speed_ms` (20ms–2000ms), effective immediately |
| Apply & Restart | Rebuilds env with new params, resets loop |

---

## Parameters mapped from `run_the_game.py`

| Variable | Widget | Default |
|---|---|---|
| `field_size` | Spinbox 4–50 | 8 |
| `number_players` | Spinbox 1–15 | 5 |
| `max_num_food` | Spinbox 1–30 | 8 |
| `coop_mode` | Checkbutton | False |
| `max_episode_steps` | Spinbox 10–500 | 50 |
| `sight` | Spinbox 0–20 | 0 |
| `min_player_level` | Spinbox 1–5 | 1 |
| `max_player_level` | Spinbox 1–5 | 1 |
| `min_food_level` | Spinbox 1–5 | 1 |
| `max_food_level` | Spinbox 1–5 | 1 |
| `penalty` | Scale 0.0–1.0 | 0.0 |
| `normalize_reward` | Checkbutton | True |
| `observe_agent_levels` | Checkbutton | True |
| `full_info_mode` | Checkbutton | True |

`grid_observation` is always `False` (not exposed — the app uses its own renderer).
`render_mode` is always `None` (Pyglet renderer disabled, app draws its own board).

---

## File locations

```
tr_lbf_addon/
  app/
    __init__.py
    game_runner.py
    board_canvas.py
    param_panel.py
    app.py
```

Entry point: `python tr_lbf_addon/app/app.py` from the project root.

No new dependencies — uses only `tkinter` (stdlib) and packages already in `.venv`.
