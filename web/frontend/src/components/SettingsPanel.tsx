import { useState, useEffect } from "react";
import type { GameParams } from "../types/game";

interface Props {
  params: GameParams | null;
  send: (msg: Record<string, unknown>) => void;
  onClose: () => void;
}

interface FieldDef {
  label: string;
  key: keyof GameParams;
  type: "spin" | "check" | "float";
  tooltip: string;
  min?: number;
  max?: number;
  step?: number;
}

const SECTIONS: { title: string; fields: FieldDef[] }[] = [
  {
    title: "Environment",
    fields: [
      { label: "Grid size", key: "field_size", type: "spin", min: 4, max: 50, tooltip: "Width and height of the square game grid" },
      { label: "Max steps", key: "max_episode_steps", type: "spin", min: 10, max: 500, tooltip: "Maximum number of steps before the episode ends" },
      { label: "Sight", key: "sight", type: "spin", min: 0, max: 20, tooltip: "How far each agent can see. 0 means full visibility" },
      { label: "Force coop", key: "coop_mode", type: "check", tooltip: "When enabled, all food requires multiple agents to collect" },
    ],
  },
  {
    title: "Players",
    fields: [
      { label: "Count", key: "number_players", type: "spin", min: 1, max: 15, tooltip: "Number of agents in the game" },
      { label: "Min level", key: "min_player_level", type: "spin", min: 1, max: 5, tooltip: "Lowest possible level assigned to an agent" },
      { label: "Max level", key: "max_player_level", type: "spin", min: 1, max: 5, tooltip: "Highest possible level assigned to an agent" },
    ],
  },
  {
    title: "Food",
    fields: [
      { label: "Max food", key: "max_num_food", type: "spin", min: 1, max: 30, tooltip: "Maximum number of food items spawned per episode" },
      { label: "Min level", key: "min_food_level", type: "spin", min: 1, max: 5, tooltip: "Lowest possible food level. Agents need combined level >= food level to collect" },
      { label: "Max level", key: "max_food_level", type: "spin", min: 1, max: 5, tooltip: "Highest possible food level" },
    ],
  },
  {
    title: "Advanced",
    fields: [
      { label: "Normalize reward", key: "normalize_reward", type: "check", tooltip: "Scale rewards by total food value so they sum to 1.0" },
      { label: "Observe levels", key: "observe_agent_levels", type: "check", tooltip: "Include agent levels in the observation space" },
      { label: "Closest fallback", key: "fallback_to_closest", type: "check", tooltip: "When no optimal fruit is found, fall back to the closest fruit instead of staying idle" },
    ],
  },
  {
    title: "Survival & Evolution",
    fields: [
      { label: "Hunger rate", key: "hunger_rate", type: "float", min: 0.0001, max: 0.1, step: 0.0001, tooltip: "Hunger increase per step. At 1.0 agent dies. Default = 1/max_steps" },
      { label: "Crowding radius", key: "crowding_radius", type: "spin", min: 1, max: 15, tooltip: "Tile radius in which nearby agents add crowding hunger penalty" },
      { label: "Crowding penalty", key: "crowding_penalty", type: "float", min: 0, max: 0.01, step: 0.0001, tooltip: "Extra hunger per nearby agent per step" },
      { label: "Food growth rate", key: "food_growth_rate", type: "float", min: 0.001, max: 1.0, step: 0.001, tooltip: "How fast food grows on grass cells (0 = never, 1 = instant)" },
      { label: "Foods per child", key: "foods_per_child", type: "spin", min: 1, max: 20, tooltip: "Number of foods an agent must eat to produce one child next round" },
      { label: "Grass ratio", key: "grass_ratio", type: "float", min: 0.1, max: 1.0, step: 0.05, tooltip: "Fraction of cells that are grass on the CA map (applied on restart)" },
      { label: "CA smoothing", key: "ca_smooth_iterations", type: "spin", min: 0, max: 20, tooltip: "CA map smoothing passes (more = larger, more connected grass areas)" },
    ],
  },
];

export default function SettingsPanel({ params, send, onClose }: Props) {
  const [local, setLocal] = useState<Record<string, unknown>>({});

  useEffect(() => {
    if (params) setLocal({ ...params });
  }, [params]);

  const setValue = (key: string, value: unknown) => {
    setLocal((prev) => ({ ...prev, [key]: value }));
  };

  const handleApply = () => {
    send({ type: "apply_params", params: { ...local } });
  };

  if (!params) return null;

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h3>Settings</h3>
        <button className="settings-close" onClick={onClose}>&times;</button>
      </div>
      <div className="settings-body">
        {SECTIONS.map((section) => (
          <fieldset key={section.title} className="settings-section">
            <legend>{section.title}</legend>
            {section.fields.map((field) => (
              <div key={field.key} className="settings-row" title={field.tooltip}>
                <label>{field.label}</label>
                {(field.type === "spin" || field.type === "float") && (
                  <input
                    type="number"
                    min={field.min}
                    max={field.max}
                    step={field.type === "float" ? (field.step ?? 0.01) : 1}
                    value={Number(local[field.key] ?? 0)}
                    onChange={(e) => setValue(field.key, Number(e.target.value))}
                  />
                )}
                {field.type === "check" && (
                  <input
                    type="checkbox"
                    checked={Boolean(local[field.key])}
                    onChange={(e) => setValue(field.key, e.target.checked)}
                  />
                )}
              </div>
            ))}
          </fieldset>
        ))}
      </div>
      <div className="settings-footer">
        <button className="btn-apply" onClick={handleApply}>Apply &amp; Restart</button>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
