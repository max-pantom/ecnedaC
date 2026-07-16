import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cadence.timeline.models import CreativeTimeline, TimelineEvent, timeline_json_schema


def event(**overrides: object) -> TimelineEvent:
    values: dict[str, object] = {
        "t_ms": 100,
        "type": "silence",
        "duration_ms": 200,
        "intensity": 0.0,
        "confidence": 0.9,
        "rationale": "Let the visual breathe",
    }
    values.update(overrides)
    return TimelineEvent.model_validate(values)


def test_valid_timeline_round_trip() -> None:
    timeline = CreativeTimeline(
        timeline_id=uuid4(),
        source_video="asset-1",
        duration_ms=1000,
        events=(event(),),
        schema_version="0.1.0",
    )
    assert CreativeTimeline.model_validate_json(timeline.model_dump_json()) == timeline
    assert timeline.events[0].type == "silence"


def test_events_must_be_ordered_and_bounded() -> None:
    with pytest.raises(ValidationError, match="ordered"):
        CreativeTimeline(
            timeline_id=uuid4(),
            source_video="asset",
            duration_ms=1000,
            events=(event(t_ms=500), event(t_ms=100)),
            schema_version="0.1.0",
        )
    with pytest.raises(ValidationError, match="beyond"):
        CreativeTimeline(
            timeline_id=uuid4(),
            source_video="asset",
            duration_ms=200,
            events=(event(t_ms=100, duration_ms=101),),
            schema_version="0.1.0",
        )


@pytest.mark.parametrize(
    "field,value",
    [("confidence", 1.1), ("intensity", -0.1), ("rationale", ""), ("type", "Bad Type")],
)
def test_event_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        event(**{field: value})


def test_json_schema_and_typescript_contain_contract_fields() -> None:
    with open("schema/timeline.schema.json", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    assert schema == timeline_json_schema()
    assert set(schema["required"]) == {
        "timeline_id", "source_video", "duration_ms", "events", "schema_version"
    }
    event_required = set(schema["$defs"]["TimelineEvent"]["required"])
    assert event_required == {"t_ms", "type", "duration_ms", "intensity", "confidence", "rationale"}
    with open("types/timeline.ts", encoding="utf-8") as type_file:
        typescript = type_file.read()
    for field in event_required:
        assert f"{field}:" in typescript
