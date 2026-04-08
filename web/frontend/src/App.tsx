import { useState, useCallback } from "react";
import { useGameSocket } from "./hooks/useGameSocket";
import { useMetricsHistory } from "./hooks/useMetricsHistory";
import Toolbar from "./components/Toolbar";
import GameBoard from "./components/GameBoard";
import SettingsPanel from "./components/SettingsPanel";
import MetricsPanel from "./components/charts/MetricsPanel";
import StatusBar from "./components/StatusBar";
import "./App.css";

export default function App() {
  const { frame, lastMessage, connected, send } = useGameSocket();
  const metrics = useMetricsHistory(lastMessage);
  const [showSettings, setShowSettings] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);
  const handleFit = useCallback(() => {
    // fit handled by GameBoard internally
  }, []);

  return (
    <div className="app">
      <Toolbar
        frame={frame}
        connected={connected}
        send={send}
        onFit={handleFit}
        onToggleSettings={() => setShowSettings((s) => !s)}
        metricsOpen={metricsOpen}
        onToggleMetrics={() => setMetricsOpen((v) => !v)}
      />
      <div className="main-area">
        <GameBoard frame={frame} />
        {showSettings && (
          <SettingsPanel
            params={frame?.params ?? null}
            send={send}
            onClose={() => setShowSettings(false)}
          />
        )}
      </div>
      <div className={`metrics-drawer ${metricsOpen ? "" : "collapsed"}`}>
        <MetricsPanel metrics={metrics} onClose={() => setMetricsOpen(false)} />
      </div>
      <StatusBar frame={frame} />
    </div>
  );
}
