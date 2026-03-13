"""Settings popup -- a tk.Toplevel with grouped parameter widgets."""

import tkinter as tk
from tkinter import ttk
from tr_lbf_addon.app.game_runner import default_params


class ParamPanel(tk.Toplevel):
    """Floating non-modal settings popup.

    Args:
        master: parent Tk root
        params: initial parameter dict (not mutated)
        on_apply: callback(params: dict) called when user clicks Apply & Restart
    """

    def __init__(self, master, params: dict, on_apply):
        super().__init__(master)
        self.title("Settings")
        self.resizable(False, False)
        self._on_apply = on_apply
        self._vars: dict[str, tk.Variable] = {}
        self._readonly_keys: set[str] = set()
        self._build(dict(params))

    # -- public ----------------------------------------------------------------

    def load_params(self, params: dict) -> None:
        """Update all widgets to reflect a new params dict."""
        for key, var in self._vars.items():
            if key in params:
                var.set(params[key])

    # -- private: layout -------------------------------------------------------

    def _build(self, params: dict) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        self._section(outer, "Environment", [
            ("Grid size",  "field_size",        "spin", 4,  50,  params),
            ("Max steps",  "max_episode_steps", "spin", 10, 500, params),
            ("Sight",      "sight",             "spin", 0,  20,  params),
            ("Force coop", "coop_mode",         "check", None, None, params),
        ])
        self._section(outer, "Players", [
            ("Count",      "number_players",    "spin", 1, 15, params),
            ("Min level",  "min_player_level",  "spin", 1,  5, params),
            ("Max level",  "max_player_level",  "spin", 1,  5, params),
        ])
        self._section(outer, "Food", [
            ("Max food",   "max_num_food",      "spin", 1, 30, params),
            ("Min level",  "min_food_level",    "spin", 1,  5, params),
            ("Max level",  "max_food_level",    "spin", 1,  5, params),
        ])
        self._section(outer, "Advanced", [
            ("Penalty",          "penalty",             "readonly", None, None, params),
            ("Normalize reward", "normalize_reward",    "check",    None, None, params),
            ("Observe levels",   "observe_agent_levels", "check",   None, None, params),
            ("Full info mode",   "full_info_mode",      "readonly", None, None, params),
            ("Closest fallback", "fallback_to_closest", "check",    None, None, params),
        ])

        btn_frame = ttk.Frame(outer)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_frame, text="Apply & Restart", command=self._apply).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Close", command=self.withdraw).pack(side=tk.LEFT)

    def _section(self, parent, title: str, rows: list) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.pack(fill=tk.X, pady=(0, 6))
        for label, key, widget_type, lo, hi, params in rows:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=18, anchor="w").pack(side=tk.LEFT)
            if widget_type == "spin":
                var = tk.IntVar(value=int(params[key]))
                ttk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=6).pack(side=tk.LEFT)
            elif widget_type == "check":
                var = tk.BooleanVar(value=bool(params[key]))
                ttk.Checkbutton(row, variable=var).pack(side=tk.LEFT)
            elif widget_type == "scale":
                var = tk.DoubleVar(value=float(params[key]))
                frame_s = ttk.Frame(row)
                frame_s.pack(side=tk.LEFT)
                ttk.Scale(frame_s, from_=lo, to=hi, variable=var,
                          orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT)
                ttk.Label(frame_s, textvariable=var, width=4).pack(side=tk.LEFT)
            elif widget_type == "readonly":
                var = tk.StringVar(value=str(params[key]))
                ttk.Label(row, textvariable=var, foreground="gray").pack(side=tk.LEFT)
                self._readonly_keys.add(key)
            else:
                var = tk.StringVar(value=str(params[key]))
            self._vars[key] = var

    def _apply(self) -> None:
        """Read all widget values and call the on_apply callback."""
        params = default_params()
        for key, var in self._vars.items():
            if key in self._readonly_keys:
                continue
            params[key] = var.get()
        # Ensure int types for spinbox values
        for int_key in ("field_size", "number_players", "max_num_food",
                        "max_episode_steps", "sight",
                        "min_player_level", "max_player_level",
                        "min_food_level", "max_food_level"):
            params[int_key] = int(params[int_key])
        self._on_apply(params)
