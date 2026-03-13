import { useState, useEffect } from "react";
import type { GameParams } from "../types/game";

interface Props {
  params: GameParams | null;
  send: (msg: Record<string, unknown>) => void;
  onClose: () => void;
}

const READONLY_KEYS = new Set(["penalty", "full_info_mode"]);

interface FieldDef {
  label: string;
  key: keyof GameParams;
  type: "spin" | "check" | "readonly";
  min?: number;
  max?: number;
}

const SECTIONS: { title: string; fields: FieldDef[] }[] = [
  {
    title: "Environment",
    fields: [
      { label: "Grid size", key: "field_size", type: "spin", min: 4, max: 50 },
      { label: "Max steps", key: "max_episode_steps", type: "spin", min: 10, max: 500 },
      { label: "Sight", key: "sight", type: "spin", min: 0, max: 20 },
      { label: "Force coop", key: "coop_mode", type: "check" },
    ],
  },
  {
    title: "Players",
    fields: [
      { label: "Count", key: "number_players", type: "spin", min: 1, max: 15 },
      { label: "Min level", key: "min_player_level", type: "spin", min: 1, max: 5 },
      { label: "Max level", key: "max_player_level", type: "spin", min: 1, max: 5 },
    ],
  },
  {
    title: "Food",
    fields: [
      { label: "Max food", key: "max_num_food", type: "spin", min: 1, max: 30 },
      { label: "Min level", key: "min_food_level", type: "spin", min: 1, max: 5 },
      { label: "Max level", key: "max_food_level", type: "spin", min: 1, max: 5 },
    ],
  },
  {
    title: "Advanced",
    fields: [
      { label: "Penalty", key: "penalty", type: "readonly" },
      { label: "Normalize reward", key: "normalize_reward", type: "check" },
      { label: "Observe levels", key: "observe_agent_levels", type: "check" },
      { label: "Full info mode", key: "full_info_mode", type: "readonly" },
      { label: "Closest fallback", key: "fallback_to_closest", type: "check" },
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
    const cleaned: Record<string, unknown> = { ...local };
    for (const key of READONLY_KEYS) {
      delete cleaned[key];
    }
    send({ type: "apply_params", params: cleaned });
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
              <div key={field.key} className="settings-row">
                <label>{field.label}</label>
                {field.type === "spin" && (
                  <input
                    type="number"
                    min={field.min}
                    max={field.max}
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
                {field.type === "readonly" && (
                  <span className="readonly-value">{String(local[field.key] ?? "")}</span>
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
