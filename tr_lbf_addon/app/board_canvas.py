import os
import math


def effective_cell_size(canvas_px: float, field_size: int, zoom: float) -> float:
    """Return the pixel size of one grid cell after applying zoom."""
    return (canvas_px / field_size) * zoom


def world_to_canvas(row: int, col: int, cell_size: float, pan_x: float, pan_y: float) -> tuple[float, float]:
    """Convert grid (row, col) to canvas pixel (x, y)."""
    return col * cell_size + pan_x, row * cell_size + pan_y


def canvas_to_world(x: float, y: float, cell_size: float, pan_x: float, pan_y: float) -> tuple[int, int]:
    """Convert canvas pixel (x, y) to grid (row, col)."""
    col = int((x - pan_x) / cell_size)
    row = int((y - pan_y) / cell_size)
    return row, col


def detail_level(effective_cell_px: float) -> str:
    """Return rendering detail level based on effective cell size.

    Returns:
        'full'    -- cell >= 40px: shapes + all text labels
        'shapes'  -- cell >= 20px: shapes only, no text
        'minimal' -- cell <  20px: agents as dots, fruits as filled cells
    """
    if effective_cell_px >= 40.0:
        return "full"
    if effective_cell_px >= 20.0:
        return "shapes"
    return "minimal"


import tkinter as tk

# Resolve icon paths relative to the project's lbforaging package
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ICON_DIR = os.path.join(_PROJECT_ROOT, "lbforaging", "foraging", "icons")
_AGENT_ICON_PATH = os.path.join(_ICON_DIR, "agent.png")
_APPLE_ICON_PATH = os.path.join(_ICON_DIR, "apple.png")

# 10-colour palette for agents
_AGENT_COLOURS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A4C93", "#1982C4", "#8AC926", "#FF595E", "#6A0572",
]
_FREE_SLOT_COLOUR = "#CCCCCC"
_GRID_COLOUR = "#DDDDDD"
_BADGE_BG = "white"
_BADGE_OUTLINE = "black"
_CANVAS_PX = 600


def _scale_photo(source: tk.PhotoImage, target_size: int) -> tk.PhotoImage:
    """Scale a tk.PhotoImage to approximately target_size using zoom + subsample.

    Uses integer zoom/subsample to get close to the desired size.
    """
    src_size = source.width()
    if src_size <= 0 or target_size <= 0:
        return source

    # Calculate best integer zoom and subsample factors
    # We want: (src_size * zoom_factor) / subsample_factor ≈ target_size
    # Try zoom first, then subsample to get closest match
    if target_size >= src_size:
        zoom_factor = max(1, round(target_size / src_size))
        subsample_factor = max(1, round((src_size * zoom_factor) / target_size))
    else:
        zoom_factor = 1
        subsample_factor = max(1, round(src_size / target_size))

    result = source
    if zoom_factor > 1:
        result = result.zoom(zoom_factor)
    if subsample_factor > 1:
        result = result.subsample(subsample_factor)
    return result


class BoardCanvas(tk.Canvas):
    """Tkinter Canvas that draws the LBF game board using original sprites.

    Uses agent.png (person silhouette) and apple.png from the lbforaging icons.
    Level badges are drawn as white circles with the level number, matching
    the original LBF rendering style.

    Call draw(runner) after each game step to redraw.
    Supports mouse-wheel zoom (centred on cursor) and click-drag pan.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, width=_CANVAS_PX, height=_CANVAS_PX,
                         bg="white", **kwargs)
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self._drag_start: tuple[float, float] | None = None
        self._drag_pan_start: tuple[float, float] | None = None

        # Load source sprite images
        self._agent_src = tk.PhotoImage(file=_AGENT_ICON_PATH)
        self._apple_src = tk.PhotoImage(file=_APPLE_ICON_PATH)

        # Cache for scaled PhotoImage references (prevents GC collection)
        self._photo_refs: list[tk.PhotoImage] = []

        self.bind("<MouseWheel>", self._on_mousewheel)        # Windows
        self.bind("<Button-4>", self._on_mousewheel)          # Linux scroll up
        self.bind("<Button-5>", self._on_mousewheel)          # Linux scroll down
        self.bind("<ButtonPress-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag)

    # -- public ----------------------------------------------------------------

    def draw(self, runner) -> None:
        """Redraw the board from the current runner state."""
        self.delete("all")
        self._photo_refs.clear()
        if runner.lbf_gym is None:
            return
        field_size = runner.params["field_size"]
        cell = effective_cell_size(_CANVAS_PX, field_size, self.zoom)
        dlevel = detail_level(cell)
        self._draw_grid(field_size, cell)
        self._draw_free_slots(runner.lbf_gym.fruits, cell, dlevel)
        self._draw_fruits(runner.lbf_gym.fruits, cell, dlevel)
        if cell >= 30:
            self._draw_target_arrows(runner.lbf_gym.agents, cell)
        self._draw_agents(runner.lbf_gym.agents, cell, dlevel)

    def reset_view(self) -> None:
        """Reset zoom and pan to fit the entire board."""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    # -- private: sprite helpers -----------------------------------------------

    def _get_scaled_sprite(self, source: tk.PhotoImage, target_size: int) -> tk.PhotoImage:
        """Scale a sprite and keep a reference to prevent garbage collection."""
        photo = _scale_photo(source, target_size)
        self._photo_refs.append(photo)
        return photo

    def _draw_level_badge(self, cell_x: float, cell_y: float, level: int, cell: float) -> None:
        """Draw a circular level badge at bottom-right of cell (original LBF style)."""
        badge_radius = cell / 5
        badge_cx = cell_x + cell * 0.75
        badge_cy = cell_y + cell * 0.25
        self.create_oval(
            badge_cx - badge_radius, badge_cy - badge_radius,
            badge_cx + badge_radius, badge_cy + badge_radius,
            fill=_BADGE_BG, outline=_BADGE_OUTLINE, width=1,
        )
        self.create_text(
            badge_cx, badge_cy,
            text=str(int(level)),
            font=("Times New Roman", max(7, int(badge_radius * 1.2)), "bold"),
            fill="black",
        )

    # -- private: zoom/pan -----------------------------------------------------

    def _on_mousewheel(self, event) -> None:
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1

        cx, cy = float(event.x), float(event.y)
        field_size_guess = max(1, int(_CANVAS_PX / max(1.0, effective_cell_size(_CANVAS_PX, 1, self.zoom))))
        cell = effective_cell_size(_CANVAS_PX, max(field_size_guess, 1), self.zoom)
        world_col = (cx - self.pan_x) / cell
        world_row = (cy - self.pan_y) / cell

        factor = 1.1 if delta > 0 else 1.0 / 1.1
        self.zoom = max(0.5, min(10.0, self.zoom * factor))

        new_cell = effective_cell_size(_CANVAS_PX, max(field_size_guess, 1), self.zoom)
        self.pan_x = cx - world_col * new_cell
        self.pan_y = cy - world_row * new_cell

    def _on_drag_start(self, event) -> None:
        self._drag_start = (float(event.x), float(event.y))
        self._drag_pan_start = (self.pan_x, self.pan_y)

    def _on_drag(self, event) -> None:
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self.pan_x = self._drag_pan_start[0] + dx
        self.pan_y = self._drag_pan_start[1] + dy

    # -- private: drawing ------------------------------------------------------

    def _draw_grid(self, field_size: int, cell: float) -> None:
        for i in range(field_size + 1):
            x = i * cell + self.pan_x
            y = i * cell + self.pan_y
            self.create_line(x, self.pan_y, x, field_size * cell + self.pan_y,
                             fill=_GRID_COLOUR, width=1)
            self.create_line(self.pan_x, y, field_size * cell + self.pan_x, y,
                             fill=_GRID_COLOUR, width=1)

    def _draw_free_slots(self, fruits, cell: float, dlevel: str) -> None:
        if dlevel == "minimal":
            return
        r = max(2.0, cell * 0.08)
        for fruit in fruits:
            for slot in (fruit.free_slots or []):
                cx, cy = world_to_canvas(slot[0], slot[1], cell, self.pan_x, self.pan_y)
                cx += cell / 2
                cy += cell / 2
                self.create_oval(cx - r, cy - r, cx + r, cy + r,
                                 fill=_FREE_SLOT_COLOUR, outline="")

    def _draw_fruits(self, fruits, cell: float, dlevel: str) -> None:
        sprite_size = max(4, int(cell * 0.8))
        for fruit in fruits:
            r, c = int(fruit.position[0]), int(fruit.position[1])
            x0, y0 = world_to_canvas(r, c, cell, self.pan_x, self.pan_y)

            if dlevel == "minimal":
                pad = cell * 0.2
                self.create_oval(x0 + pad, y0 + pad, x0 + cell - pad, y0 + cell - pad,
                                 fill="#D32F2F", outline="")
            else:
                photo = self._get_scaled_sprite(self._apple_src, sprite_size)
                self.create_image(x0 + cell / 2, y0 + cell / 2, image=photo, anchor=tk.CENTER)
                if dlevel == "full":
                    self._draw_level_badge(x0, y0, fruit.level, cell)

    def _draw_target_arrows(self, agents, cell: float) -> None:
        for agent in agents:
            if agent.target is None:
                continue
            ax, ay = world_to_canvas(int(agent.position[0]), int(agent.position[1]),
                                     cell, self.pan_x, self.pan_y)
            fx, fy = world_to_canvas(int(agent.target.position[0]),
                                     int(agent.target.position[1]),
                                     cell, self.pan_x, self.pan_y)
            colour = _AGENT_COLOURS[agent.id % len(_AGENT_COLOURS)]
            self.create_line(ax + cell / 2, ay + cell / 2,
                             fx + cell / 2, fy + cell / 2,
                             fill=colour, dash=(4, 3), width=1, arrow=tk.LAST)

    def _draw_agents(self, agents, cell: float, dlevel: str) -> None:
        sprite_size = max(4, int(cell * 0.8))
        for agent in agents:
            colour = _AGENT_COLOURS[agent.id % len(_AGENT_COLOURS)]
            r, c = int(agent.position[0]), int(agent.position[1])
            x0, y0 = world_to_canvas(r, c, cell, self.pan_x, self.pan_y)

            if dlevel == "minimal":
                dot_r = max(2.0, cell * 0.3)
                cx, cy = x0 + cell / 2, y0 + cell / 2
                self.create_oval(cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
                                 fill=colour, outline="")
            else:
                # Draw coloured background square to tint the agent
                pad = cell * 0.1
                self.create_rectangle(x0 + pad, y0 + pad,
                                      x0 + cell - pad, y0 + cell - pad,
                                      fill=colour, outline="", width=0)
                # Overlay the agent silhouette sprite
                photo = self._get_scaled_sprite(self._agent_src, sprite_size)
                self.create_image(x0 + cell / 2, y0 + cell / 2, image=photo, anchor=tk.CENTER)
                if dlevel == "full":
                    self._draw_level_badge(x0, y0, agent.level, cell)
