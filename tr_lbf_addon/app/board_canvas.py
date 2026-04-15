"""Viewport and coordinate utilities for the game board canvas.

Python port of web/frontend/src/lib/viewport.ts. Used by the Tkinter board
canvas to map between world (row, col) coordinates and canvas pixel positions.
"""

from typing import Literal

DetailLevel = Literal["full", "shapes", "minimal"]


def effective_cell_size(canvas_px: float, field_size: int, zoom: float) -> float:
    """Compute the pixel size of one grid cell at the given zoom level.

    Args:
        canvas_px: shorter side of the canvas in pixels.
        field_size: number of cells along one axis.
        zoom: current zoom multiplier (1.0 = fit the grid to canvas).

    Returns:
        Cell size in pixels.
    """
    return (canvas_px / field_size) * zoom


def world_to_canvas(
    row: int,
    col: int,
    cell_size: float,
    pan_x: float,
    pan_y: float,
) -> tuple[float, float]:
    """Convert a world grid position to canvas pixel coordinates (top-left of cell).

    Args:
        row: grid row (0 = top).
        col: grid column (0 = left).
        cell_size: pixel size of one cell.
        pan_x: horizontal pan offset in pixels.
        pan_y: vertical pan offset in pixels.

    Returns:
        (x, y) pixel position of the top-left corner of the cell.
    """
    return col * cell_size + pan_x, row * cell_size + pan_y


def canvas_to_world(
    x: float,
    y: float,
    cell_size: float,
    pan_x: float,
    pan_y: float,
) -> tuple[int, int]:
    """Convert canvas pixel coordinates to the nearest world grid cell.

    Args:
        x: horizontal pixel position on the canvas.
        y: vertical pixel position on the canvas.
        cell_size: pixel size of one cell.
        pan_x: horizontal pan offset in pixels.
        pan_y: vertical pan offset in pixels.

    Returns:
        (row, col) of the grid cell that contains the given pixel.
    """
    return int((y - pan_y) / cell_size), int((x - pan_x) / cell_size)


def detail_level(effective_cell_px: float) -> DetailLevel:
    """Choose a rendering detail level based on the current cell pixel size.

    Args:
        effective_cell_px: cell size in pixels as returned by effective_cell_size.

    Returns:
        "full"    — cell >= 40 px: sprites, badges, hunger bars.
        "shapes"  — cell >= 20 px: sprites only, no badges.
        "minimal" — cell <  20 px: coloured dots only.
    """
    if effective_cell_px >= 40:
        return "full"
    if effective_cell_px >= 20:
        return "shapes"
    return "minimal"
