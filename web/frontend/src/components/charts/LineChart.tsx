import {
  ResponsiveContainer,
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { MetricSeriesData } from "../../types/metrics";

interface Props {
  series: MetricSeriesData[];
  xLabel?: string;
  yLabel?: string;
  height?: number;
  animate?: boolean;
  yScale?: "linear" | "log";
}

/**
 * Generic dark-themed line chart wrapping Recharts.
 * One <Line> per MetricSeriesData entry.
 */
export default function LineChart({
  series,
  xLabel = "Episode",
  yLabel = "Value",
  height = 240,
  animate = true,
  yScale = "linear",
}: Props) {
  // Merge all series points into a single array keyed by x value
  const pointMap = new Map<number, Record<string, number>>();
  for (const s of series) {
    for (const pt of s.points) {
      if (!pointMap.has(pt.x)) pointMap.set(pt.x, { x: pt.x });
      pointMap.get(pt.x)![s.key] = pt.y;
    }
  }
  const chartData = Array.from(pointMap.values()).sort((a, b) => a.x - b.x);

  const axisStyle = { fontSize: 11, fill: "var(--text-secondary, #aaa)" };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsLineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border, #333)" />
        <XAxis
          dataKey="x"
          label={{ value: xLabel, position: "insideBottomRight", offset: -4, style: axisStyle }}
          tick={axisStyle}
          type="number"
          domain={["dataMin", "dataMax"]}
        />
        <YAxis
          scale={yScale}
          domain={yScale === "log" ? ["auto", "auto"] : undefined}
          label={{ value: yLabel, angle: -90, position: "insideLeft", style: axisStyle }}
          tick={axisStyle}
          width={52}
        />
        <Tooltip
          contentStyle={{
            background: "var(--bg-surface, #1e1e1e)",
            border: "1px solid var(--border, #444)",
            color: "var(--text-primary, #eee)",
            fontSize: 12,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 4 }} />
        {series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.name}
            stroke={s.color}
            dot={false}
            isAnimationActive={animate}
            strokeWidth={2}
          />
        ))}
      </RechartsLineChart>
    </ResponsiveContainer>
  );
}
