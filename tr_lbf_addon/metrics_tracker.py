"""Tracks learning metrics across episodes for plotting."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MetricSeries:
    """A named time-series of (x, y) metric points with a rolling window."""

    name: str
    unit: str
    max_points: int = 2000
    points: list[tuple[float, float]] = field(default_factory=list)

    def append(self, x: float, y: float) -> None:
        """Append a point and drop the oldest if the window is full."""
        self.points.append((x, y))
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points :]

    def clear(self) -> None:
        self.points = []

    def to_dict(self) -> dict:
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
        self.episode_index: int = 0
        self.series: dict[str, MetricSeries] = {
            "episode_return_mean": MetricSeries("Mean Episode Return", "reward"),
            "episode_return_total": MetricSeries("Total Episode Return", "reward"),
            "nn_loss_mean": MetricSeries("Mean NN Loss", "MSE"),
        }
        # Keyed by agent id; per-agent series created lazily
        self._pending_losses_per_agent: dict[int, list[float]] = {}
        self._dirty: bool = False

    # -- recording API ---------------------------------------------------------

    def record_step_losses(self, losses_per_agent: dict[int, list[float]]) -> None:
        """Accumulate NN losses for this step into the episode buffer."""
        for agent_id, losses in losses_per_agent.items():
            if losses:
                if agent_id not in self._pending_losses_per_agent:
                    self._pending_losses_per_agent[agent_id] = []
                self._pending_losses_per_agent[agent_id].extend(losses)

    def record_episode_end(self, cumulative_rewards: list[float]) -> None:
        """Finalise one episode: compute statistics and append one point per series."""
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
        """Full snapshot — all series with all points."""
        return {
            "episode_index": self.episode_index,
            "series": {key: s.to_dict() for key, s in self.series.items()},
        }

    def latest_values(self) -> dict | None:
        """Latest point per series if a new episode completed since last call, else None."""
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
        """Full reset — called on rebuild / apply_params."""
        self.episode_index = 0
        for s in self.series.values():
            s.clear()
        # Preserve lazily-created per-agent series but wipe their data
        self._pending_losses_per_agent = {}
        self._dirty = False
