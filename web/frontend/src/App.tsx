import { useState, useCallback, useRef } from "react";
import { useGameSocket } from "./hooks/useGameSocket";
import Toolbar from "./components/Toolbar";
import GameBoard from "./components/GameBoard";
import SettingsPanel from "./components/SettingsPanel";
import StatusBar from "./components/StatusBar";
import "./App.css";

export default function App() {
  const { frame, connected, send } = useGameSocket();
  const [showSettings, setShowSettings] = useState(false);
  const boardRef = useRef<{ resetView: () => void }>(null);

  const handleFit = useCallback(() => {
    boardRef.current?.resetView();
  }, []);

  return (
    <div className="app">
      <Toolbar
        frame={frame}
        connected={connected}
        send={send}
        onFit={handleFit}
        onToggleSettings={() => setShowSettings((s) => !s)}
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
      <StatusBar frame={frame} />
    </div>
  );
}
