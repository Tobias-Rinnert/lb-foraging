export interface AgentState {
  id: number;
  position: [number, number];
  level: number;
  color: string;
  target_position: [number, number] | null;
  is_loading: boolean;
  is_alive?: boolean;
  hunger?: number;
  food_eaten?: number;
  nn_architecture?: { embedding_dim: number; decision_hidden: number };
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
  // Survival & evolution
  hunger_rate: number;
  food_growth_rate: number;
  foods_per_child: number;
  grass_ratio: number;
  ca_smooth_iterations: number;
}

export interface MetricsLatest {
  type: "metrics_latest";
  episode_index: number;
  values: Record<string, number>;
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
  metrics_latest?: MetricsLatest;
  ca_map?: number[][] | null;
  food_growth?: Record<string, number>;
  dead_agents?: number[];
  population_size?: number;
  next_population_size?: number;
}
