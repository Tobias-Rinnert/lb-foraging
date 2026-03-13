# tr_lbf_addon/tests/test_app_board_canvas.py
import numpy as np
from tr_lbf_addon.app.board_canvas import (
    effective_cell_size,
    world_to_canvas,
    canvas_to_world,
    detail_level,
)


def test_effective_cell_size_no_zoom():
    assert effective_cell_size(canvas_px=600, field_size=8, zoom=1.0) == 75.0


def test_effective_cell_size_with_zoom():
    assert effective_cell_size(canvas_px=600, field_size=8, zoom=2.0) == 150.0


def test_world_to_canvas_origin():
    x, y = world_to_canvas(row=0, col=0, cell_size=75.0, pan_x=0.0, pan_y=0.0)
    assert x == 0.0 and y == 0.0


def test_world_to_canvas_offset():
    # row=1, col=2 with cell_size=75, no pan -> x=150, y=75
    x, y = world_to_canvas(row=1, col=2, cell_size=75.0, pan_x=0.0, pan_y=0.0)
    assert x == 150.0 and y == 75.0


def test_world_to_canvas_with_pan():
    x, y = world_to_canvas(row=0, col=0, cell_size=75.0, pan_x=10.0, pan_y=20.0)
    assert x == 10.0 and y == 20.0


def test_canvas_to_world_roundtrip():
    row, col = canvas_to_world(x=150.0, y=75.0, cell_size=75.0, pan_x=0.0, pan_y=0.0)
    assert row == 1 and col == 2


def test_detail_level_full():
    assert detail_level(40.0) == "full"


def test_detail_level_shapes():
    assert detail_level(25.0) == "shapes"


def test_detail_level_minimal():
    assert detail_level(15.0) == "minimal"


def test_detail_level_boundary_40():
    assert detail_level(39.9) == "shapes"


def test_detail_level_boundary_20():
    assert detail_level(19.9) == "minimal"


def test_board_canvas_importable():
    """BoardCanvas can be imported without a display."""
    from tr_lbf_addon.app import board_canvas  # noqa: F401
