"""Contained media lookup and bounded byte-range streaming."""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from cadence.dataset.records import SegmentCandidate, SourceRecord

MAX_RANGE_BYTES = 8 * 1024 * 1024
STREAM_CHUNK_BYTES = 64 * 1024
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class RegistryLookup(Protocol):
    def get_source(self, source_id: UUID | str) -> SourceRecord: ...

    def get_segment(self, segment_id: UUID | str) -> SegmentCandidate: ...


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int
    total: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def resolve_registered_media(
    registry: RegistryLookup, entity_id: str, intake_root: Path
) -> Path:
    """Resolve an opaque registry ID, never a caller-controlled filesystem path."""
    record: object
    candidates: tuple[str, ...]
    try:
        record = registry.get_source(entity_id)
        candidates = ("normalized_path", "storage_path")
    except (KeyError, ValueError):
        try:
            record = registry.get_segment(entity_id)
            candidates = ("extracted_path",)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown media entity") from exc

    unresolved = next(
        (
            Path(value)
            for field in candidates
            if (value := getattr(record, field, None)) is not None
        ),
        None,
    )
    if unresolved is None:
        raise HTTPException(status_code=404, detail="media is unavailable")

    root = intake_root.resolve()
    try:
        resolved = unresolved.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail="media is unavailable") from exc
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise HTTPException(status_code=403, detail="media path is outside the intake root")
    return resolved


def media_response(path: Path, range_header: str | None) -> StreamingResponse:
    total = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": "inline",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, no-store",
    }
    if range_header is None:
        headers["Content-Length"] = str(total)
        return StreamingResponse(
            _read_chunks(path, 0, total),
            media_type=content_type,
            headers=headers,
        )

    selected = parse_range(range_header, total)
    headers.update(
        {
            "Content-Length": str(selected.length),
            "Content-Range": f"bytes {selected.start}-{selected.end}/{selected.total}",
        }
    )
    return StreamingResponse(
        _read_chunks(path, selected.start, selected.length),
        status_code=206,
        media_type=content_type,
        headers=headers,
    )


def parse_range(value: str, total: int) -> ByteRange:
    match = _RANGE.fullmatch(value.strip())
    if match is None or total <= 0:
        raise HTTPException(
            status_code=416,
            detail="invalid media range",
            headers={"Content-Range": f"bytes */{total}"},
        )
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        raise HTTPException(
            status_code=416,
            detail="invalid media range",
            headers={"Content-Range": f"bytes */{total}"},
        )
    if not start_text:
        requested = min(int(end_text), MAX_RANGE_BYTES, total)
        if requested <= 0:
            raise HTTPException(
                status_code=416,
                detail="unsatisfiable media range",
                headers={"Content-Range": f"bytes */{total}"},
            )
        start, end = total - requested, total - 1
    else:
        start = int(start_text)
        requested_end = int(end_text) if end_text else total - 1
        if start >= total or requested_end < start:
            raise HTTPException(
                status_code=416,
                detail="unsatisfiable media range",
                headers={"Content-Range": f"bytes */{total}"},
            )
        end = min(requested_end, total - 1, start + MAX_RANGE_BYTES - 1)
    return ByteRange(start=start, end=end, total=total)


def _read_chunks(path: Path, start: int, length: int) -> Iterator[bytes]:
    remaining = length
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining:
            chunk = handle.read(min(STREAM_CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
