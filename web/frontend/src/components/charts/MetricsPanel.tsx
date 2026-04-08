import type { MetricsState } from "../../types/metrics";
import LearningChart from "./LearningChart";

interface Props {
  metrics: MetricsState;
  onClose: () => void;
}

/**
 * Bottom-drawer container holding the learning metrics charts.
 * Add additional <XxxChart> components here as the analysis suite grows.
 */
export default function MetricsPanel({ metrics, onClose }: Props) {
  return (
    <div className="metrics-panel">
      <div className="metrics-panel-header">
        <span style={{ fontWeight: 600, fontSize: 13 }}>
          Learning Metrics — Episode {metrics.episodeIndex}
        </span>
        <button onClick={onClose} style={{ padding: "2px 8px" }}>✕</button>
      </div>
      <div className="chart-container">
        <LearningChart metrics={metrics} />
      </div>
    </div>
  );
}
