"""LBF Game App -- entry point.

Run from the project root:
    python tr_lbf_addon/app/app.py
"""
import sys
import os

# Ensure tr_lbf_addon/ is on sys.path for bare imports in lbf_gym.py
_addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _addon_dir not in sys.path:
    sys.path.insert(0, _addon_dir)

# Ensure project root is on sys.path for lbforaging package
_project_root = os.path.dirname(_addon_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tkinter as tk
from tkinter import ttk

from tr_lbf_addon.app.game_runner import GameRunner, default_params
from tr_lbf_addon.app.board_canvas import BoardCanvas
from tr_lbf_addon.app.param_panel import ParamPanel

_MIN_SPEED_MS = 20
_MAX_SPEED_MS = 2000
_DEFAULT_SPEED_MS = 200


class LBFApp:
    """Root application controller."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LBF Foraging")
        self.root.resizable(True, True)

        self._runner = GameRunner(default_params())
        self._paused = True
        self._speed_ms = _DEFAULT_SPEED_MS
        self._after_id = None
        self._param_panel: ParamPanel | None = None

        self._build_toolbar()
        self._build_canvas()
        self._build_status_bar()

        self._runner.reset()
        self._redraw()

    # -- public ----------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()

    # -- toolbar ---------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Play",  width=8, command=self._play).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Pause", width=8, command=self._pause).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Step",  width=8, command=self._manual_step).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        ttk.Label(bar, text="Speed:").pack(side=tk.LEFT)
        self._speed_var = tk.IntVar(value=_DEFAULT_SPEED_MS)
        ttk.Scale(bar, from_=_MIN_SPEED_MS, to=_MAX_SPEED_MS,
                  variable=self._speed_var, orient=tk.HORIZONTAL, length=120,
                  command=lambda _: self._update_speed()).pack(side=tk.LEFT, padx=4)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        ttk.Button(bar, text="Fit", width=6, command=self._fit).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Settings", command=self._open_settings).pack(side=tk.RIGHT, padx=2)

        self._step_label = ttk.Label(bar, text="Step 0/0")
        self._step_label.pack(side=tk.RIGHT, padx=8)

    def _build_canvas(self) -> None:
        self._canvas = BoardCanvas(self.root)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        # Bind canvas mouse events to trigger redraw after interaction
        self._canvas.bind("<MouseWheel>", lambda e: (self._canvas._on_mousewheel(e), self._redraw()), add=True)
        self._canvas.bind("<Button-4>",   lambda e: (self._canvas._on_mousewheel(e), self._redraw()), add=True)
        self._canvas.bind("<Button-5>",   lambda e: (self._canvas._on_mousewheel(e), self._redraw()), add=True)
        self._canvas.bind("<B1-Motion>",  lambda e: (self._canvas._on_drag(e),       self._redraw()), add=True)

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2)
                  ).pack(side=tk.BOTTOM, fill=tk.X)

    # -- controls --------------------------------------------------------------

    def _play(self) -> None:
        if not self._paused:
            return
        self._paused = False
        self._schedule_step()

    def _pause(self) -> None:
        self._paused = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _manual_step(self) -> None:
        self._do_step()

    def _fit(self) -> None:
        self._canvas.reset_view()
        self._redraw()

    def _update_speed(self) -> None:
        self._speed_ms = int(self._speed_var.get())

    def _open_settings(self) -> None:
        if self._param_panel is None or not self._param_panel.winfo_exists():
            self._param_panel = ParamPanel(
                self.root,
                self._runner.params,
                on_apply=self._apply_params,
            )
        else:
            self._param_panel.deiconify()
            self._param_panel.lift()

    def _apply_params(self, new_params: dict) -> None:
        self._pause()
        self._runner.rebuild(new_params)
        self._canvas.reset_view()
        self._redraw()

    # -- game loop -------------------------------------------------------------

    def _schedule_step(self) -> None:
        self._after_id = self.root.after(self._speed_ms, self._auto_step)

    def _auto_step(self) -> None:
        if self._paused:
            return
        self._do_step()
        if not self._runner.episode_over:
            self._schedule_step()

    def _do_step(self) -> None:
        self._runner.step()
        self._redraw()
        if self._runner.episode_over:
            self._show_game_over()

    def _redraw(self) -> None:
        self._canvas.draw(self._runner)
        p = self._runner.params
        self._step_label.config(
            text=f"Step {self._runner.step_count}/{p['max_episode_steps']}"
        )
        rewards_str = "  ".join(
            f"A{i}: {r:.2f}" for i, r in enumerate(self._runner.rewards)
        )
        status = "Paused" if self._paused else "Running"
        self._status_var.set(f"{status}  |  {rewards_str}")

    def _show_game_over(self) -> None:
        self._paused = True
        total = sum(self._runner.rewards)
        self._status_var.set(f"Episode over -- total reward: {total:.2f}  |  Click Play to restart")
        # Auto-reset for next episode
        self._runner.reset()


def main() -> None:
    app = LBFApp()
    app.run()


if __name__ == "__main__":
    main()
