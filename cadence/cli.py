"""Cadence local-readiness command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cadence.common.config import load_config


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cadence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_check = subparsers.add_parser("config-check")
    config_check.add_argument("--config", required=True)

    data_policy = subparsers.add_parser("data-policy")
    data_policy_commands = data_policy.add_subparsers(dest="data_policy_command", required=True)
    data_policy_check = data_policy_commands.add_parser("check")
    data_policy_check.add_argument("--repo-root", default=".")

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
    remote.add_argument(
        "action",
        choices=[
            "bootstrap_vps",
            "doctor_vps",
            "submit_job",
            "sync_checkpoints",
            "fetch_results",
            "terminate_gpu",
        ],
    )
    remote.add_argument("--config", default="configs/vps.yaml")
    remote.add_argument("--execute", action="store_true")
    from cadence.dataset.cli import add_dataset_parsers
    from cadence.ingestion.pilot_cli import add_pilot_parsers

    add_dataset_parsers(subparsers)
    add_pilot_parsers(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config-check":
        _json(load_config(args.config).model_dump(mode="json"))
    elif args.command == "data-policy":
        from cadence.common.data_policy import check_repository_data_policy

        report = check_repository_data_policy(args.repo_root)
        _json(report.to_dict())
        return 0 if report.passed else 1
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
        _json(
            {
                "video_parameters": count_parameters(video),
                "audio_parameters": count_parameters(audio),
                "video_memory": estimate_training_memory(
                    video,
                    (
                        config.runtime.microbatch_size,
                        3,
                        config.data.num_frames,
                        config.data.frame_size,
                        config.data.frame_size,
                    ),
                ).__dict__,
            }
        )
    elif args.command == "train-synthetic":
        from cadence.training.synthetic import run_synthetic_training

        config = load_config(args.config)
        _json(
            run_synthetic_training(
                config, checkpoint_path=args.checkpoint, resume_from=args.resume_from
            )
        )
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

        _json(run_contrastive_training(load_config(args.config), resume_from=args.resume_from))
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
    elif args.command in {"dataset", "storage"}:
        from cadence.dataset.cli import handle_dataset_command

        _json(handle_dataset_command(args))
    elif args.command == "pilot":
        from cadence.ingestion.pilot_cli import handle_pilot_command

        _json(handle_pilot_command(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
