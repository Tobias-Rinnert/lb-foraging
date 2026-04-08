"""Convert MetricsTracker state into JSON-serializable dicts for WebSocket messages."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tr_lbf_addon.metrics_tracker import MetricsTracker


def serialize_metrics_snapshot(tracker: "MetricsTracker") -> dict:
    """Full snapshot of all metric series — sent on connect and after rebuild."""
    snap = tracker.snapshot()
    return {
        "type": "metrics_snapshot",
        "episode_index": snap["episode_index"],
        "series": snap["series"],
    }


def serialize_metrics_latest(tracker: "MetricsTracker") -> dict | None:
    """Latest episode values if a new episode completed since last call, else None."""
    latest = tracker.latest_values()
    if latest is None:
        return None
    return {
        "type": "metrics_latest",
        "episode_index": latest["episode_index"],
        "values": latest["values"],
    }
