import { useState, useEffect } from "react";
import type { MetricsState, MetricSeriesData, MetricPoint } from "../types/metrics";
import { getAgentColor } from "../constants/agentColors";

const SERIES_COLORS: Record<string, string> = {
  episode_return_mean: "var(--accent, #E63946)",
  episode_return_total: "#457B9D",
  nn_loss_mean: "#2A9D8F",
};

function colorForKey(key: string): string {
  if (key in SERIES_COLORS) return SERIES_COLORS[key];
  const match = key.match(/nn_loss_agent_(\d+)/);
  if (match) return getAgentColor(parseInt(match[1], 10));
  return "#888888";
}

const EMPTY_STATE: MetricsState = { episodeIndex: 0, series: {} };

/**
 * Maintains a rolling history of metric series from WebSocket messages.
 *
 * Handles three message types:
 *   - metrics_snapshot: replace entire state
 *   - metrics_cleared:  reset to empty
 *   - frame with metrics_latest: append one point per series
 */
export function useMetricsHistory(
  lastMessage: unknown,
  maxPointsPerSeries = 2000,
): MetricsState {
  const [state, setState] = useState<MetricsState>(EMPTY_STATE);

  useEffect(() => {
    if (!lastMessage || typeof lastMessage !== "object") return;
    const msg = lastMessage as Record<string, unknown>;

    if (msg["type"] === "metrics_snapshot") {
      // Full replace
      const rawSeries = (msg["series"] ?? {}) as Record<
        string,
        { name: string; unit: string; points: MetricPoint[] }
      >;
      const series: Record<string, MetricSeriesData> = {};
      for (const [key, raw] of Object.entries(rawSeries)) {
        series[key] = {
          key,
          name: raw.name,
          unit: raw.unit,
          color: colorForKey(key),
          points: raw.points.slice(-maxPointsPerSeries),
        };
      }
      setState({ episodeIndex: (msg["episode_index"] as number) ?? 0, series });
      return;
    }

    if (msg["type"] === "metrics_cleared") {
      setState(EMPTY_STATE);
      return;
    }

    // Game frame — check for embedded metrics_latest
    const ml = msg["metrics_latest"] as
      | { episode_index: number; values: Record<string, number> }
      | undefined;
    if (!ml) return;

    setState((prev) => {
      const episode_index = ml.episode_index;
      const updated = { ...prev.series };
      for (const [key, value] of Object.entries(ml.values)) {
        const existing = updated[key];
        const newPoint: MetricPoint = { x: episode_index, y: value };
        if (existing) {
          updated[key] = {
            ...existing,
            points: [...existing.points, newPoint].slice(-maxPointsPerSeries),
          };
        } else {
          // Series not yet in state (edge case: snapshot not yet received)
          updated[key] = {
            key,
            name: key,
            unit: "",
            color: colorForKey(key),
            points: [newPoint],
          };
        }
      }
      return { episodeIndex: episode_index + 1, series: updated };
    });
  }, [lastMessage, maxPointsPerSeries]);

  return state;
}
