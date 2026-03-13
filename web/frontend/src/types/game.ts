export interface AgentState {
  id: number;
  position: [number, number];
  level: number;
  color: string;
  target_position: [number, number] | null;
  is_loading: boolean;
}

export interface FruitState {
  position: [number, number];
  level: number;
  free_slots: [number, number][];
}

export interface GameParams {
  field_size: number;
  number_players: number;
  max_num_food: number;
  coop_mode: boolean;
  max_episode_steps: number;
  sight: number;
  min_player_level: number;
  max_player_level: number;
  min_food_level: number;
  max_food_level: number;
  penalty: number;
  normalize_reward: boolean;
  observe_agent_levels: boolean;
  full_info_mode: boolean;
  fallback_to_closest: boolean;
}

export interface GameFrame {
  field_size: number;
  step_count: number;
  max_steps: number;
  episode_over: boolean;
  paused: boolean;
  rewards: number[];
  agents: AgentState[];
  fruits: FruitState[];
  params: GameParams;
}
