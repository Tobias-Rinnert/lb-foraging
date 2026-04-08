/**
 * Agent colour palette — mirrors _AGENT_COLOURS in web/backend/serializer.py.
 * Canonical definition lives in the Python file; keep both in sync.
 */
const AGENT_COLOURS: string[] = [
  "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
  "#6A4C93", "#1982C4", "#8AC926", "#FF595E", "#6A0572",
];

export function getAgentColor(id: number): string {
  return AGENT_COLOURS[id % AGENT_COLOURS.length];
}
