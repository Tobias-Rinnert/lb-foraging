"""Convert GameRunner state into JSON-serializable dicts."""

import numpy as np
from web.backend.metrics_serializer import serialize_metrics_latest

# Same 10-colour palette as board_canvas.py
_AGENT_COLOURS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A4C93", "#1982C4", "#8AC926", "#FF595E", "#6A0572",
]


def _to_list(val):
    """Convert numpy arrays / scalars to plain Python types."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def serialize_frame(runner, paused: bool) -> dict:
    """Serialize the current GameRunner state to a JSON-ready dict."""
    agents = []
    fruits = []

    if runner.lbf_gym is not None:
        for agent in runner.lbf_gym.agents:
            target_pos = None
            if agent.target is not None:
                target_pos = _to_list(agent.target.position)
            agents.append({
                "id": int(agent.id),
                "position": _to_list(agent.position),
                "level": int(agent.level),
                "color": _AGENT_COLOURS[agent.id % len(_AGENT_COLOURS)],
                "target_position": target_pos,
                "is_loading": bool(agent.is_loading),
                "is_alive": bool(getattr(agent, "is_alive", True)),
                "hunger": float(runner.agent_hunger.get(agent.id, 0.0)),
                "food_eaten": int(runner.agent_food_eaten.get(agent.id, 0)),
                "nn_architecture": {
                    "embedding_dim": int(getattr(agent, "embedding_dim", 8)),
                    "decision_hidden": int(getattr(agent, "decision_hidden", 16)),
                },
            })

        for fruit in runner.lbf_gym.fruits:
            free_slots = []
            if fruit.free_slots is not None:
                free_slots = [_to_list(s) for s in fruit.free_slots]
            fruits.append({
                "position": _to_list(fruit.position),
                "level": int(fruit.level),
                "free_slots": free_slots,
            })

    frame = {
        "field_size": int(runner.params["field_size"]),
        "step_count": int(runner.step_count),
        "max_steps": int(runner.params["max_episode_steps"]),
        "episode_over": bool(runner.episode_over),
        "paused": paused,
        "rewards": [float(r) for r in runner.rewards],
        "agents": agents,
        "fruits": fruits,
        "params": {k: _to_list(v) for k, v in runner.params.items() if not k.startswith("_")},
        "ca_map": runner.ca_map.tolist() if runner.ca_map is not None else None,
        "food_growth": {
            f"{k[0]},{k[1]}": round(float(v), 4)
            for k, v in runner.food_growth.items()
        },
        "dead_agents": list(runner.dead_agents),
        "population_size": len(runner.lbf_gym.agents) if runner.lbf_gym else 0,
        "next_population_size": runner._next_player_count,
    }
    metrics_latest = serialize_metrics_latest(runner.metrics)
    if metrics_latest is not None:
        frame["metrics_latest"] = metrics_latest
    return frame
