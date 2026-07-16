# Creative Timeline specification

The canonical implementation is `cadence.timeline.models`; `schema/timeline.schema.json` is the
wire contract. Events are ordered, remain within the timeline, require rationale and confidence,
and use an extensible lowercase event type. Silence is represented by an event whose type is
`silence`.

