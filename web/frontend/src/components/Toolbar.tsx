import { useState } from "react";
import type { GameFrame } from "../types/game";

interface Props {
  frame: GameFrame | null;
  connected: boolean;
  send: (msg: Record<string, unknown>) => void;
  onFit: () => void;
  onToggleSettings: () => void;
  metricsOpen: boolean;
  onToggleMetrics: () => void;
}

export default function Toolbar({ frame, connected, send, onFit, onToggleSettings, metricsOpen, onToggleMetrics }: Props) {
  const [speed, setSpeed] = useState(200);

  const handleSpeedChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = Number(e.target.value);
    setSpeed(val);
    send({ type: "set_speed", speed_ms: val });
  };

  return (
    <div className="toolbar">
      <div className="toolbar-group">
        <button onClick={() => send({ type: "play" })} disabled={!connected} title="Play">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <polygon points="3,1 13,8 3,15" />
          </svg>
        </button>
        <button onClick={() => send({ type: "pause" })} disabled={!connected} title="Pause">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <rect x="3" y="1" width="3.5" height="14" />
            <rect x="9.5" y="1" width="3.5" height="14" />
          </svg>
        </button>
        <button onClick={() => send({ type: "step" })} disabled={!connected} title="Step">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <polygon points="2,1 10,8 2,15" />
            <rect x="11" y="1" width="3" height="14" />
          </svg>
        </button>
      </div>

      <div className="toolbar-separator" />

      <div className="toolbar-group">
        <label className="speed-label">
          Speed
          <input
            type="range"
            min={20}
            max={2000}
            value={speed}
            onChange={handleSpeedChange}
            className="speed-slider"
          />
          <span className="speed-value">{speed}ms</span>
        </label>
      </div>

      <div className="toolbar-separator" />

      <div className="toolbar-group">
        <button onClick={onFit} title="Fit to view">Fit</button>
      </div>

      <div className="toolbar-spacer" />

      <div className="toolbar-group">
        <span className="step-counter">
          Step {frame?.step_count ?? 0}/{frame?.max_steps ?? 0}
        </span>
      </div>

      <div className="toolbar-group">
        <span className={`connection-dot ${connected ? "connected" : "disconnected"}`} />
        <button onClick={onToggleMetrics} style={{ fontWeight: metricsOpen ? 600 : undefined }}>
          Metrics
        </button>
        <button onClick={onToggleSettings}>Settings</button>
      </div>
    </div>
  );
}
