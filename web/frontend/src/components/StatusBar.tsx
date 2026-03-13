import type { GameFrame } from "../types/game";

interface Props {
  frame: GameFrame | null;
}

export default function StatusBar({ frame }: Props) {
  if (!frame) {
    return <div className="status-bar">Connecting...</div>;
  }

  let status: string;
  if (frame.episode_over) {
    const total = frame.rewards.reduce((a, b) => a + b, 0);
    status = `Episode over - total reward: ${total.toFixed(2)}`;
  } else if (frame.paused) {
    status = "Paused";
  } else {
    status = "Running";
  }

  const rewardsStr = frame.rewards
    .map((r, i) => `A${i}: ${r.toFixed(2)}`)
    .join("  ");

  return (
    <div className="status-bar">
      <span className="status-state">{status}</span>
      <span className="status-rewards">{rewardsStr}</span>
    </div>
  );
}
