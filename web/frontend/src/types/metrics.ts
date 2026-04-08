export interface MetricPoint {
  x: number;
  y: number;
}

export interface MetricSeriesData {
  key: string;
  name: string;
  unit: string;
  color: string;
  points: MetricPoint[];
}

export interface MetricsState {
  episodeIndex: number;
  series: Record<string, MetricSeriesData>;
}
