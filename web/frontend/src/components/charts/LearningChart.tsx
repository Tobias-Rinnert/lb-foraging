import { useState } from "react";
import type { MetricsState } from "../../types/metrics";
import LineChart from "./LineChart";

const DEFAULT_VISIBLE = new Set(["episode_return_mean", "nn_loss_mean"]);

interface Props {
  metrics: MetricsState;
}

/**
 * Learning-curve chart with per-series visibility toggles and a log-scale option.
 */
export default function LearningChart({ metrics }: Props) {
  const [visible, setVisible] = useState<Set<string>>(DEFAULT_VISIBLE);
  const [logScale, setLogScale] = useState(false);

  const allKeys = Object.keys(metrics.series);
  const selectedSeries = allKeys
    .filter((k) => visible.has(k))
    .map((k) => metrics.series[k]);

  const toggleKey = (key: string) => {
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="chart-toggles">
        {allKeys.map((key) => {
          const s = metrics.series[key];
          return (
            <label key={key} style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={visible.has(key)}
                onChange={() => toggleKey(key)}
              />
              <span style={{ color: s.color }}>{s.name}</span>
            </label>
          );
        })}
        <label style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
          <input type="checkbox" checked={logScale} onChange={() => setLogScale((v) => !v)} />
          Log scale
        </label>
      </div>
      <div className="chart-container">
        {selectedSeries.length > 0 ? (
          <LineChart
            series={selectedSeries}
            xLabel="Episode"
            yLabel="Value"
            animate={false}
            yScale={logScale ? "log" : "linear"}
          />
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-secondary, #888)", fontSize: 13 }}>
            No series selected
          </div>
        )}
      </div>
    </div>
  );
}
