"""Tests for MetricsTracker and MetricSeries."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics_tracker import MetricSeries, MetricsTracker


def test_metric_series_append():
    series = MetricSeries("Test", "unit")
    series.append(0.0, 1.5)
    assert series.points == [(0.0, 1.5)]


def test_metric_series_rolling_window_drops_oldest():
    series = MetricSeries("Test", "unit", max_points=5)
    for i in range(10):
        series.append(float(i), float(i))
    assert len(series.points) == 5
    assert series.points[0][0] == 5.0   # oldest kept is x=5
    assert series.points[-1][0] == 9.0  # newest is x=9


def test_tracker_record_episode_end_advances_index():
    tracker = MetricsTracker()
    assert tracker.episode_index == 0
    tracker.record_episode_end([1.0, 2.0])
    assert tracker.episode_index == 1
    tracker.record_episode_end([0.5])
    assert tracker.episode_index == 2


def test_tracker_record_episode_end_computes_mean_and_total_return():
    tracker = MetricsTracker()
    tracker.record_episode_end([1.0, 3.0])
    mean_pts = tracker.series["episode_return_mean"].points
    total_pts = tracker.series["episode_return_total"].points
    assert mean_pts[0] == (0.0, 2.0)
    assert total_pts[0] == (0.0, 4.0)


def test_tracker_loss_aggregation_mean():
    tracker = MetricsTracker()
    tracker.record_step_losses({0: [0.1, 0.2], 1: [0.3]})
    tracker.record_episode_end([1.0])
    # overall mean = (0.1 + 0.2 + 0.3) / 3
    overall = tracker.series["nn_loss_mean"].points[0][1]
    assert abs(overall - 0.2) < 1e-6
    # per-agent means
    assert abs(tracker.series["nn_loss_agent_0"].points[0][1] - 0.15) < 1e-6
    assert abs(tracker.series["nn_loss_agent_1"].points[0][1] - 0.3) < 1e-6


def test_tracker_clear_resets_episode_and_series():
    tracker = MetricsTracker()
    tracker.record_episode_end([1.0, 2.0])
    tracker.clear()
    assert tracker.episode_index == 0
    for s in tracker.series.values():
        assert s.points == []


def test_tracker_latest_values_dirty_flag():
    tracker = MetricsTracker()
    assert tracker.latest_values() is None  # nothing recorded yet
    tracker.record_episode_end([1.0])
    result = tracker.latest_values()
    assert result is not None
    assert "episode_return_mean" in result["values"]
    # second call should return None (flag cleared)
    assert tracker.latest_values() is None


def test_tracker_snapshot_round_trip_json_serializable():
    tracker = MetricsTracker()
    tracker.record_step_losses({0: [0.5]})
    tracker.record_episode_end([1.0, 2.0])
    snapshot = tracker.snapshot()
    serialized = json.dumps(snapshot)  # must not raise
    parsed = json.loads(serialized)
    assert parsed["episode_index"] == 1
    assert "episode_return_mean" in parsed["series"]


def test_tracker_per_agent_series_created_lazily():
    tracker = MetricsTracker()
    assert "nn_loss_agent_0" not in tracker.series
    tracker.record_step_losses({0: [0.1]})
    tracker.record_episode_end([1.0])
    assert "nn_loss_agent_0" in tracker.series
