"""Compatibility CLI for the branch-local launch-video pilot workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any
from uuid import UUID

DEFAULT_PILOT_DIR = "data/pilots/launch-video"


def add_pilot_parsers(subparsers: Any) -> None:
    """Register the research-pilot commands without shadowing dataset intake."""
    pilot = subparsers.add_parser(
        "pilot",
        help="run the legacy launch-video research pilot workflow",
    )
    pilot_sub = pilot.add_subparsers(dest="pilot_command", required=True)

    pilot_source = pilot_sub.add_parser("source")
    source_sub = pilot_source.add_subparsers(dest="source_command", required=True)
    source_add = source_sub.add_parser("add")
    source_add.add_argument("urls", nargs="*")
    source_add.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    source_add.add_argument("--media-path")
    source_add.add_argument("--source-url")
    source_add.add_argument("--creator")
    source_add.add_argument("--submitted-by", default="user")
    source_add.add_argument("--collection-method", default="user-submitted-url")
    source_add.add_argument("--license-status", default="unverified-research-quarantine")
    source_inspect = source_sub.add_parser("inspect")
    source_inspect.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    source_inspect.add_argument("--source", default="all")
    source_approve = source_sub.add_parser("approve")
    source_approve.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    source_approve.add_argument("--source", action="append", required=True)
    source_download = source_sub.add_parser("download")
    source_download.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    source_download.add_argument("--source", action="append", required=True)
    source_reject = source_sub.add_parser("reject")
    source_reject.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    source_reject.add_argument("--source", action="append", required=True)
    source_reject.add_argument("--reason", required=True)

    pilot_segments = pilot_sub.add_parser("segments")
    segments_sub = pilot_segments.add_subparsers(dest="segments_command", required=True)
    segments_suggest = segments_sub.add_parser("suggest")
    segments_suggest.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    segments_suggest.add_argument("--source", required=True)
    segments_suggest.add_argument("--min-duration", type=float, default=4.0)
    segments_suggest.add_argument("--max-duration", type=float, default=10.0)
    segments_approve = segments_sub.add_parser("approve")
    segments_approve.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    segments_approve.add_argument("--clip", action="append", required=True)
    segments_reject = segments_sub.add_parser("reject")
    segments_reject.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    segments_reject.add_argument("--clip", action="append", required=True)
    segments_reject.add_argument("--reason", required=True)

    pilot_build = pilot_sub.add_parser("build")
    pilot_build.add_argument("dataset_id")
    pilot_build.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)

    pilot_report = pilot_sub.add_parser("report")
    pilot_report.add_argument("dataset_id")
    pilot_report.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)


def _selected_ids(values: list[str], available: list[UUID]) -> list[UUID]:
    return available if values == ["all"] else [UUID(value) for value in values]


def handle_pilot_command(args: argparse.Namespace) -> object:
    """Execute one launch-video pilot command."""
    from cadence.ingestion.dataset_pilot import (
        _read_segments,
        _read_sources,
        approve_segments,
        approve_sources,
        build_pilot_manifest,
        build_report,
        download_sources,
        inspect_source,
        reject_segments,
        reject_sources,
        suggest_segments,
        write_candidate_sources,
        write_source_record,
    )

    if args.pilot_command == "source" and args.source_command == "add":
        if args.media_path:
            if not args.source_url:
                raise ValueError("--source-url is required when --media-path is provided")
            return asdict(
                write_source_record(
                    args.pilot_dir,
                    media_path=args.media_path,
                    source_url=args.source_url,
                    creator=args.creator,
                    collection_method=args.collection_method,
                    license_status=args.license_status,
                )
            )
        urls = list(args.urls)
        if args.source_url:
            urls.append(args.source_url)
        if not urls:
            raise ValueError("provide one or more URLs, or --media-path with --source-url")
        return [
            asdict(source)
            for source in write_candidate_sources(
                args.pilot_dir,
                urls,
                submitted_by=args.submitted_by,
                collection_method=args.collection_method,
            )
        ]

    if args.pilot_command == "source" and args.source_command == "inspect":
        sources = _read_sources(args.pilot_dir)
        if args.source != "all":
            source_id = UUID(args.source)
            sources = [source for source in sources if source.source_asset_id == source_id]
        return [asdict(inspect_source(source)) for source in sources]

    if args.pilot_command == "source" and args.source_command in {
        "approve",
        "download",
        "reject",
    }:
        source_ids = _selected_ids(
            args.source,
            [source.source_asset_id for source in _read_sources(args.pilot_dir)],
        )
        if args.source_command == "approve":
            return [asdict(source) for source in approve_sources(args.pilot_dir, source_ids)]
        if args.source_command == "download":
            return [asdict(source) for source in download_sources(args.pilot_dir, source_ids)]
        return [
            asdict(source)
            for source in reject_sources(args.pilot_dir, source_ids, reason=args.reason)
        ]

    if args.pilot_command == "segments" and args.segments_command == "suggest":
        sources = _read_sources(args.pilot_dir)
        selected = (
            sources
            if args.source == "all"
            else [source for source in sources if source.source_asset_id == UUID(args.source)]
        )
        candidates = []
        for source in selected:
            if source.media_path is not None:
                candidates.extend(
                    suggest_segments(
                        args.pilot_dir,
                        source.source_asset_id,
                        min_duration_s=args.min_duration,
                        max_duration_s=args.max_duration,
                    )
                )
        return [asdict(candidate) for candidate in candidates]

    if args.pilot_command == "segments" and args.segments_command in {
        "approve",
        "reject",
    }:
        clip_ids = _selected_ids(
            args.clip,
            [segment.clip_asset_id for segment in _read_segments(args.pilot_dir)],
        )
        if args.segments_command == "approve":
            return [asdict(segment) for segment in approve_segments(args.pilot_dir, clip_ids)]
        return [
            asdict(segment)
            for segment in reject_segments(args.pilot_dir, clip_ids, reason=args.reason)
        ]

    if args.pilot_command == "build":
        return {"manifest": str(build_pilot_manifest(args.pilot_dir, dataset_id=args.dataset_id))}
    if args.pilot_command == "report":
        return build_report(args.pilot_dir, dataset_id=args.dataset_id)
    raise ValueError("unsupported pilot command")
