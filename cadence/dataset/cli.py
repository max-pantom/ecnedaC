"""Nested dataset-intake and storage command-line interface."""

from __future__ import annotations

import argparse
import os

from cadence.common.config import load_config
from cadence.dataset.downloaders import DirectHTTPDownloader, DownloaderChain, YtDlpDownloader
from cadence.dataset.media import FFmpegMediaProcessor
from cadence.dataset.records import ApprovalStatus, RightsStatus
from cadence.dataset.service import GIB, DatasetIntakeService


def add_dataset_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--config", default="configs/vps.yaml")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)

    source = dataset_commands.add_parser("source")
    source_commands = source.add_subparsers(dest="source_command", required=True)
    add = source_commands.add_parser("add")
    add.add_argument("url")
    add.add_argument("--submitted-by", default=os.getenv("USER", "unknown-operator"))
    batch = source_commands.add_parser("add-batch")
    batch.add_argument("file")
    batch.add_argument("--submitted-by", default=os.getenv("USER", "unknown-operator"))
    source_commands.add_parser("list")
    source_id_commands = (
        "inspect",
        "approve",
        "reject",
        "approve-download",
        "reject-download",
        "download",
    )
    for command in source_id_commands:
        action = source_commands.add_parser(command)
        action.add_argument("source_id")
    rights = source_commands.add_parser("rights")
    rights.add_argument("source_id")
    rights.add_argument("--status", required=True, choices=[item.value for item in RightsStatus])
    rights.add_argument("--notes", required=True)
    eligibility = source_commands.add_parser("eligibility")
    eligibility.add_argument("source_id")
    eligibility_group = eligibility.add_mutually_exclusive_group(required=True)
    eligibility_group.add_argument("--eligible", action="store_true")
    eligibility_group.add_argument("--ineligible", action="store_true")

    segments = dataset_commands.add_parser("segments")
    segment_list = segments.add_subparsers(dest="segments_command", required=True)
    for command in ("suggest", "list"):
        action = segment_list.add_parser(command)
        action.add_argument("source_id")

    segment = dataset_commands.add_parser("segment")
    segment_commands = segment.add_subparsers(dest="segment_command", required=True)
    for command in ("approve", "reject"):
        action = segment_commands.add_parser(command)
        action.add_argument("segment_id")

    build = dataset_commands.add_parser("build")
    build.add_argument("dataset_name")
    report = dataset_commands.add_parser("report")
    report.add_argument("dataset_name")
    legacy_import = dataset_commands.add_parser(
        "legacy-import",
        help="quarantine source records from the retired cadence pilot registry",
    )
    legacy_import.add_argument("pilot_dir")
    legacy_import.add_argument(
        "--submitted-by",
        default=os.getenv("USER", "unknown-operator"),
    )
    legacy_import.add_argument("--execute", action="store_true")

    storage = subparsers.add_parser("storage")
    storage.add_argument("--config", default="configs/vps.yaml")
    storage_commands = storage.add_subparsers(dest="storage_command", required=True)
    storage_commands.add_parser("report")


def build_service(config_path: str) -> DatasetIntakeService:
    config = load_config(config_path)
    maximum_bytes = round(config.dataset_intake.unknown_download_reservation_gb * GIB)
    chain = DownloaderChain(
        [DirectHTTPDownloader(maximum_bytes=maximum_bytes), YtDlpDownloader()]
    )
    media = FFmpegMediaProcessor(
        config.dataset_intake.ffmpeg_binary, config.dataset_intake.ffprobe_binary
    )
    return DatasetIntakeService(config, downloaders=chain, media=media)


def handle_dataset_command(args: argparse.Namespace) -> object:
    service = build_service(args.config)
    if args.command == "storage":
        return service.storage.report().to_dict()
    if args.dataset_command == "source":
        if args.source_command == "add":
            source, created = service.add_source(args.url, submitted_by=args.submitted_by)
            return {"created": created, "source": source.model_dump(mode="json")}
        if args.source_command == "add-batch":
            return service.add_batch(args.file, submitted_by=args.submitted_by)
        if args.source_command == "list":
            return [source.model_dump(mode="json") for source in service.list_sources()]
        if args.source_command == "inspect":
            return service.inspect_source(args.source_id).model_dump(mode="json")
        if args.source_command == "approve":
            return service.set_source_approval(
                args.source_id, ApprovalStatus.APPROVED
            ).model_dump(mode="json")
        if args.source_command == "reject":
            return service.set_source_approval(
                args.source_id, ApprovalStatus.REJECTED
            ).model_dump(mode="json")
        if args.source_command == "approve-download":
            return service.set_download_approval(
                args.source_id, ApprovalStatus.APPROVED
            ).model_dump(mode="json")
        if args.source_command == "reject-download":
            return service.set_download_approval(
                args.source_id, ApprovalStatus.REJECTED
            ).model_dump(mode="json")
        if args.source_command == "rights":
            return service.set_rights(
                args.source_id, RightsStatus(args.status), license_notes=args.notes
            ).model_dump(mode="json")
        if args.source_command == "eligibility":
            return service.set_training_eligibility(
                args.source_id, bool(args.eligible)
            ).model_dump(mode="json")
        if args.source_command == "download":
            return service.download_source(args.source_id).model_dump(mode="json")
    if args.dataset_command == "segments":
        if args.segments_command == "suggest":
            return [
                segment.model_dump(mode="json")
                for segment in service.suggest_source_segments(args.source_id)
            ]
        return [
            segment.model_dump(mode="json") for segment in service.list_segments(args.source_id)
        ]
    if args.dataset_command == "segment":
        status = (
            ApprovalStatus.APPROVED
            if args.segment_command == "approve"
            else ApprovalStatus.REJECTED
        )
        return service.set_segment_approval(args.segment_id, status).model_dump(mode="json")
    if args.dataset_command == "build":
        return service.build_dataset(args.dataset_name).model_dump(mode="json")
    if args.dataset_command == "report":
        return service.dataset_report(args.dataset_name)
    if args.dataset_command == "legacy-import":
        return service.import_legacy_pilot(
            args.pilot_dir,
            submitted_by=args.submitted_by,
            execute=args.execute,
        )
    raise ValueError("unsupported dataset command")
