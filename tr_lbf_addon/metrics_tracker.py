"""Tracks learning metrics across episodes for plotting."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MetricSeries:
    """A named time-series of (x, y) metric points with a rolling window.

    Attributes:
        name: display name for the metric (e.g. "Mean Episode Return")
        unit: unit string (e.g. "reward", "MSE")
        max_points: maximum points to store; oldest dropped on overflow
        points: list of (x, y) tuples where x is episode index, y is value
    """

    name: str
    "Display name for this metric series"
    unit: str
    "Unit of measurement (e.g. 'reward', 'MSE')"
    max_points: int = 2000
    "Maximum number of points to retain in the rolling window"
    points: list[tuple[float, float]] = field(default_factory=list)
    "List of (episode_index, value) tuples"

    def append(self, x: float, y: float) -> None:
        """Append a point (x, y) and drop the oldest if window is full.

        Args:
            x: episode index or step number
            y: metric value
        """
        self.points.append((x, y))
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points :]

    def clear(self) -> None:
        """Clear all points from this series."""
        self.points = []

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict format.

        Returns:
            Dict with keys: name (str), unit (str), points (list of {x, y} dicts)
        """
        return {
            "name": self.name,
            "unit": self.unit,
            "points": [{"x": x, "y": y} for x, y in self.points],
        }


class MetricsTracker:
    """Accumulates per-step losses and per-episode returns, persisting across episodes.

    Usage pattern:
      - Each step: call record_step_losses(losses_per_agent)
      - On episode end: call record_episode_end(cumulative_rewards)
      - On rebuild / user reset: call clear()
    """

    def __init__(self) -> None:
        """Initialize metrics tracker with empty series and state.

        Creates three default series: episode_return_mean, episode_return_total, nn_loss_mean.
        Per-agent series are created lazily on first use.
        """
        self.episode_index: int = 0
        "Current episode number"
        self.series: dict[str, MetricSeries] = {
            "episode_return_mean": MetricSeries("Mean Episode Return", "reward"),
            "episode_return_total": MetricSeries("Total Episode Return", "reward"),
            "nn_loss_mean": MetricSeries("Mean NN Loss", "MSE"),
        }
        "Dict of named metric series; per-agent series keyed by agent_id"
        self._pending_losses_per_agent: dict[int, list[float]] = {}
        "Buffer for losses accumulating during current episode, keyed by agent_id"
        self._dirty: bool = False
        "True when latest_values() should recompute; cleared on read"

    # -- recording API ---------------------------------------------------------

    def record_step_losses(self, losses_per_agent: dict[int, list[float]]) -> None:
        """Accumulate NN losses for this step into the episode buffer.

        Args:
            losses_per_agent: dict mapping agent_id to list of MSE loss values
        """
        for agent_id, losses in losses_per_agent.items():
            if losses:
                if agent_id not in self._pending_losses_per_agent:
                    self._pending_losses_per_agent[agent_id] = []
                self._pending_losses_per_agent[agent_id].extend(losses)

    def record_episode_end(self, cumulative_rewards: list[float]) -> None:
        """Finalize one episode: compute statistics and append to all series.

        Computes mean and total episode rewards, mean NN loss (all agents),
        and per-agent mean loss. Increments episode_index and resets loss buffer.
        Sets _dirty flag so latest_values() recomputes.

        Args:
            cumulative_rewards: list of total rewards for each agent in this episode

        Returns:
            None (side effects on self.series, episode_index, _pending_losses_per_agent, _dirty)
        """
        x = float(self.episode_index)

        # Return series
        total_return = sum(cumulative_rewards)
        mean_return = total_return / len(cumulative_rewards) if cumulative_rewards else 0.0
        self.series["episode_return_mean"].append(x, mean_return)
        self.series["episode_return_total"].append(x, total_return)

        # Loss series — overall mean
        all_losses: list[float] = []
        for agent_id, losses in self._pending_losses_per_agent.items():
            all_losses.extend(losses)
            # Per-agent series — created lazily
            series_key = f"nn_loss_agent_{agent_id}"
            if series_key not in self.series:
                self.series[series_key] = MetricSeries(f"Agent {agent_id} NN Loss", "MSE")
            agent_mean = sum(losses) / len(losses) if losses else 0.0
            self.series[series_key].append(x, agent_mean)

        overall_mean_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
        self.series["nn_loss_mean"].append(x, overall_mean_loss)

        self._pending_losses_per_agent = {}
        self.episode_index += 1
        self._dirty = True

    # -- query API -------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a complete JSON-serializable snapshot of all series.

        Returns:
            Dict with keys: episode_index (int), series (dict of series_key→MetricSeries.to_dict())
        """
        return {
            "episode_index": self.episode_index,
            "series": {key: s.to_dict() for key, s in self.series.items()},
        }

    def latest_values(self) -> dict | None:
        """Return the most recent metric values if episode just completed, else None.

        Returns None if _dirty is False (no new episode data since last call).
        Clears _dirty flag on read.

        Returns:
            Dict with keys: episode_index (just-completed episode), values (dict of metric→float),
            or None if no new data.
        """
        if not self._dirty:
            return None
        self._dirty = False
        values: dict[str, float] = {}
        for key, s in self.series.items():
            if s.points:
                values[key] = s.points[-1][1]
        return {
            "episode_index": self.episode_index - 1,
            "values": values,
        }

    def clear(self) -> None:
        """Reset all data for a fresh tracking session.

        Clears all series data, resets episode_index to 0, clears loss buffer.
        Called on rebuild() or parameter change.

        Returns:
            None (side effects on all internal state)
        """
        self.episode_index = 0
        for s in self.series.values():
            s.clear()
        # Preserve lazily-created per-agent series but wipe their data
        self._pending_losses_per_agent = {}
        self._dirty = False
