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
