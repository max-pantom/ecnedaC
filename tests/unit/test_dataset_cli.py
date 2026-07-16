from cadence.cli import build_parser


def test_required_dataset_cli_shapes_parse() -> None:
    parser = build_parser()
    commands = [
        ["dataset", "source", "add", "https://example.com/launch.mp4"],
        ["dataset", "source", "add-batch", "urls.txt"],
        ["dataset", "source", "list"],
        ["dataset", "source", "inspect", "00000000-0000-0000-0000-000000000000"],
        ["dataset", "source", "approve", "00000000-0000-0000-0000-000000000000"],
        ["dataset", "source", "reject", "00000000-0000-0000-0000-000000000000"],
        ["dataset", "source", "download", "00000000-0000-0000-0000-000000000000"],
        ["dataset", "segments", "suggest", "00000000-0000-0000-0000-000000000000"],
        ["dataset", "segments", "list", "00000000-0000-0000-0000-000000000000"],
        ["dataset", "segment", "approve", "00000000-0000-0000-0000-000000000000"],
        ["dataset", "segment", "reject", "00000000-0000-0000-0000-000000000000"],
        ["dataset", "build", "launch-pilot"],
        ["dataset", "report", "launch-pilot"],
        ["storage", "report"],
    ]
    for command in commands:
        assert parser.parse_args(command).command in {"dataset", "storage"}

