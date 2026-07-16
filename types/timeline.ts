// Mirrors schema/timeline.schema.json. The JSON Schema and Python models are canonical.
export type TimelineEvent = {
  t_ms: number;
  type: string;
  duration_ms: number;
  intensity: number;
  confidence: number;
  rationale: string;
  texture?: string | null;
  pan?: number | null;
  reverb?: string | null;
};

export type CreativeTimeline = {
  timeline_id: string;
  source_video: string;
  duration_ms: number;
  events: TimelineEvent[];
  schema_version: "0.1.0";
};

