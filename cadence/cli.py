"""Cadence local-readiness command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import UUID

from cadence.common.config import load_config


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cadence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_check = subparsers.add_parser("config-check")
    config_check.add_argument("--config", required=True)

    fixture = subparsers.add_parser("fixture-generate")
    fixture.add_argument("--output-dir", required=True)

    manifest = subparsers.add_parser("manifest-validate")
    manifest.add_argument("manifest")

    inspect_model = subparsers.add_parser("model-inspect")
    inspect_model.add_argument("--config", required=True)

    synthetic = subparsers.add_parser("train-synthetic")
    synthetic.add_argument("--config", required=True)
    synthetic.add_argument("--checkpoint", default="artifacts/checkpoints/synthetic.pt")
    synthetic.add_argument("--resume-from")

    train = subparsers.add_parser("train-contrastive")
    train.add_argument("--config", required=True)
    train.add_argument("--resume-from")

    retrieval = subparsers.add_parser("retrieval-eval")
    retrieval.add_argument("--config", required=True)
    retrieval.add_argument("--synthetic", action="store_true", required=True)

    checkpoint = subparsers.add_parser("checkpoint-inspect")
    checkpoint.add_argument("checkpoint")

    package = subparsers.add_parser("remote-package")
    package.add_argument("--config", required=True)
    package.add_argument("--output", default="artifacts/reports/remote-job.json")
    package.add_argument("--allow-dirty", action="store_true")

    remote = subparsers.add_parser("remote-action")
    remote.add_argument("action", choices=[
        "bootstrap_vps", "doctor_vps", "submit_job", "sync_checkpoints",
        "fetch_results", "terminate_gpu",
    ])
    remote.add_argument("--config", default="configs/vps.yaml")
    remote.add_argument("--execute", action="store_true")

    dataset = subparsers.add_parser("dataset")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    dataset_source = dataset_sub.add_parser("source")
    source_sub = dataset_source.add_subparsers(dest="source_command", required=True)
    source_add = source_sub.add_parser("add")
    source_add.add_argument("urls", nargs="*")
    source_add.add_argument("--pilot-dir", default="data/pilots/launch-video")
    source_add.add_argument("--media-path")
    source_add.add_argument("--source-url")
    source_add.add_argument("--creator")
    source_add.add_argument("--submitted-by", default="user")
    source_add.add_argument("--collection-method", default="user-submitted-url")
    source_add.add_argument("--license-status", default="unverified-research-quarantine")
    source_inspect = source_sub.add_parser("inspect")
    source_inspect.add_argument("--pilot-dir", default="data/pilots/launch-video")
    source_inspect.add_argument("--source", default="all")
    source_approve = source_sub.add_parser("approve")
    source_approve.add_argument("--pilot-dir", default="data/pilots/launch-video")
    source_approve.add_argument("--source", action="append", required=True)
    source_download = source_sub.add_parser("download")
    source_download.add_argument("--pilot-dir", default="data/pilots/launch-video")
    source_download.add_argument("--source", action="append", required=True)

    dataset_segments = dataset_sub.add_parser("segments")
    segments_sub = dataset_segments.add_subparsers(dest="segments_command", required=True)
    segments_suggest = segments_sub.add_parser("suggest")
    segments_suggest.add_argument("--pilot-dir", default="data/pilots/launch-video")
    segments_suggest.add_argument("--source", required=True)
    segments_suggest.add_argument("--min-duration", type=float, default=4.0)
    segments_suggest.add_argument("--max-duration", type=float, default=10.0)
    segments_approve = segments_sub.add_parser("approve")
    segments_approve.add_argument("--pilot-dir", default="data/pilots/launch-video")
    segments_approve.add_argument("--clip", action="append", required=True)
    segments_reject = segments_sub.add_parser("reject")
    segments_reject.add_argument("--pilot-dir", default="data/pilots/launch-video")
    segments_reject.add_argument("--clip", action="append", required=True)
    segments_reject.add_argument("--reason", required=True)

    dataset_build = dataset_sub.add_parser("build")
    dataset_build.add_argument("dataset_id")
    dataset_build.add_argument("--pilot-dir", default="data/pilots/launch-video")

    dataset_report = dataset_sub.add_parser("report")
    dataset_report.add_argument("dataset_id")
    dataset_report.add_argument("--pilot-dir", default="data/pilots/launch-video")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config-check":
        _json(load_config(args.config).model_dump(mode="json"))
    elif args.command == "fixture-generate":
        from cadence.ingestion.fixtures import generate_fixtures

        _json({"manifest": str(generate_fixtures(args.output_dir))})
    elif args.command == "manifest-validate":
        from cadence.ingestion.manifest import load_manifest

        _json({"entries": len(load_manifest(args.manifest)), "valid": True})
    elif args.command == "model-inspect":
        from cadence.encoders.common import count_parameters, estimate_training_memory
        from cadence.training.synthetic import build_models

        config = load_config(args.config)
        video, audio = build_models(config)
        _json({
            "video_parameters": count_parameters(video),
            "audio_parameters": count_parameters(audio),
            "video_memory": estimate_training_memory(
                video,
                (config.runtime.microbatch_size, 3, config.data.num_frames,
                 config.data.frame_size, config.data.frame_size),
            ).__dict__,
        })
    elif args.command == "train-synthetic":
        from cadence.training.synthetic import run_synthetic_training

        config = load_config(args.config)
        _json(run_synthetic_training(
            config, checkpoint_path=args.checkpoint, resume_from=args.resume_from
        ))
    elif args.command == "retrieval-eval":
        import torch

        from cadence.training.contrastive import evaluate_retrieval

        config = load_config(args.config)
        count = max(2, config.runtime.contrastive_group_size)
        video_embedding = torch.eye(count)
        audio_embedding = torch.eye(count)
        _json(
            evaluate_retrieval(
                video_embedding, audio_embedding, config.training.temperature
            ).to_dict()
        )
    elif args.command == "train-contrastive":
        from cadence.training.runner import run_contrastive_training

        _json(
            run_contrastive_training(
                load_config(args.config), resume_from=args.resume_from
            )
        )
    elif args.command == "checkpoint-inspect":
        from cadence.training.checkpoint import inspect_checkpoint

        _json(inspect_checkpoint(args.checkpoint))
    elif args.command == "remote-package":
        from cadence.remote.job import package_remote_job, write_remote_job

        config = load_config(args.config)
        job = package_remote_job(config, require_clean=not args.allow_dirty)
        write_remote_job(job, args.output)
        _json({"output": str(Path(args.output).resolve()), "git_commit": job.git_commit})
    elif args.command == "remote-action":
        from cadence.remote.job import run_remote_action

        print(run_remote_action(args.action, load_config(args.config), execute=args.execute))
    elif args.command == "dataset":
        from cadence.ingestion.dataset_pilot import (
            approve_segments,
            approve_sources,
            build_pilot_manifest,
            build_report,
            download_sources,
            inspect_source,
            reject_segments,
            suggest_segments,
            write_source_record,
        )
        from cadence.ingestion.dataset_pilot import _read_segments, _read_sources, write_candidate_sources

        if args.dataset_command == "source" and args.source_command == "add":
            if args.media_path:
                if not args.source_url:
                    raise ValueError("--source-url is required when --media-path is provided")
                _json(write_source_record(
                    args.pilot_dir,
                    media_path=args.media_path,
                    source_url=args.source_url,
                    creator=args.creator,
                    collection_method=args.collection_method,
                    license_status=args.license_status,
                ))
            else:
                urls = list(args.urls)
                if args.source_url:
                    urls.append(args.source_url)
                if not urls:
                    raise ValueError("provide one or more URLs, or --media-path with --source-url")
                _json(write_candidate_sources(
                    args.pilot_dir,
                    urls,
                    submitted_by=args.submitted_by,
                    collection_method=args.collection_method,
                ))
        elif args.dataset_command == "source" and args.source_command == "inspect":
            sources = _read_sources(args.pilot_dir)
            if args.source != "all":
                source_id = UUID(args.source)
                sources = [source for source in sources if source.source_asset_id == source_id]
            _json([inspect_source(source) for source in sources])
        elif args.dataset_command == "source" and args.source_command == "approve":
            if args.source == ["all"]:
                source_ids = [source.source_asset_id for source in _read_sources(args.pilot_dir)]
            else:
                source_ids = [UUID(value) for value in args.source]
            _json(approve_sources(args.pilot_dir, source_ids))
        elif args.dataset_command == "source" and args.source_command == "download":
            if args.source == ["all"]:
                source_ids = [source.source_asset_id for source in _read_sources(args.pilot_dir)]
            else:
                source_ids = [UUID(value) for value in args.source]
            _json(download_sources(args.pilot_dir, source_ids))
        elif args.dataset_command == "segments" and args.segments_command == "suggest":
            sources = _read_sources(args.pilot_dir)
            selected = sources if args.source == "all" else [
                source for source in sources if source.source_asset_id == UUID(args.source)
            ]
            candidates = []
            for source in selected:
                if source.media_path is None:
                    continue
                candidates.extend(suggest_segments(
                    args.pilot_dir,
                    source.source_asset_id,
                    min_duration_s=args.min_duration,
                    max_duration_s=args.max_duration,
                ))
            _json(candidates)
        elif args.dataset_command == "segments" and args.segments_command == "approve":
            if args.clip == ["all"]:
                clip_ids = [segment.clip_asset_id for segment in _read_segments(args.pilot_dir)]
            else:
                clip_ids = [UUID(value) for value in args.clip]
            _json(approve_segments(args.pilot_dir, clip_ids))
        elif args.dataset_command == "segments" and args.segments_command == "reject":
            if args.clip == ["all"]:
                clip_ids = [segment.clip_asset_id for segment in _read_segments(args.pilot_dir)]
            else:
                clip_ids = [UUID(value) for value in args.clip]
            _json(reject_segments(args.pilot_dir, clip_ids, reason=args.reason))
        elif args.dataset_command == "build":
            _json({"manifest": str(build_pilot_manifest(args.pilot_dir, dataset_id=args.dataset_id))})
        elif args.dataset_command == "report":
            _json(build_report(args.pilot_dir, dataset_id=args.dataset_id))
        else:
            raise ValueError("unsupported dataset command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
