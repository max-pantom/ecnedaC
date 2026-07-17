from pathlib import Path
from unittest.mock import patch

from cadence.dataset.media import FFmpegMediaProcessor


def test_ffmpeg_media_processor_disables_stdin_for_unattended_operations(tmp_path: Path) -> None:
    processor = FFmpegMediaProcessor()
    source = tmp_path / "source.mp4"
    normalized = tmp_path / "normalized.mp4"
    segment = tmp_path / "segment.mp4"
    source.write_bytes(b"not-real-media")

    with patch.object(processor, "_require_binary"), patch.object(
        processor, "probe"
    ) as probe, patch("cadence.dataset.media.subprocess.run") as run:
        probe.return_value = object()
        processor.normalize(source, normalized)
        processor.extract_segment(source, segment, 0.0, 4.0)

    normalize_cmd = run.call_args_list[0].args[0]
    segment_cmd = run.call_args_list[1].args[0]
    assert "-nostdin" in normalize_cmd
    assert "-nostdin" in segment_cmd
    assert normalize_cmd.index("-nostdin") < normalize_cmd.index("-i")
    assert segment_cmd.index("-nostdin") < segment_cmd.index("-i")
