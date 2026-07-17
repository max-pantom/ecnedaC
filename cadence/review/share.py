"""Guarded, short-lived Wormkey sharing for the private review console."""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

from cadence.review.auth import configured_secret

WORMKEY_VERSION = "0.1.5"
DEFAULT_EXPIRY = "30m"
MINIMUM_EXPIRY_SECONDS = 5 * 60
MAXIMUM_EXPIRY_SECONDS = 2 * 60 * 60
_EXPIRY = re.compile(r"^([1-9][0-9]*)([mh])$")
_PUBLIC_URL = re.compile(r"^https://(?:[a-z0-9-]+\.)?wormkey\.run/s/[a-z0-9-]+$")


@dataclass(frozen=True)
class WormkeySharePlan:
    provider: Literal["wormkey"]
    package: str
    config_path: str
    local_url: str
    expires: str
    execute: bool
    command: tuple[str, ...]
    protections: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_wormkey_share_plan(
    config_path: str | Path,
    *,
    port: int = 8787,
    expires: str = DEFAULT_EXPIRY,
    execute: bool = False,
) -> WormkeySharePlan:
    if not 1 <= port <= 65535:
        raise ValueError("review share port must be between 1 and 65535")
    expiry_seconds(expires)
    package = f"wormkey@{WORMKEY_VERSION}"
    return WormkeySharePlan(
        provider="wormkey",
        package=package,
        config_path=str(Path(config_path)),
        local_url=f"http://127.0.0.1:{port}",
        expires=expires,
        execute=execute,
        command=("npx", "--yes", package, "http", str(port), "--expires", expires),
        protections=(
            "loopback-only Cadence listener",
            "Cadence-enforced ephemeral outer Basic authentication",
            "forced Secure, HttpOnly, SameSite=Strict session cookie",
            "administrator login rate limit",
            "maximum two-hour tunnel lifetime",
            "automatic server and tunnel cleanup",
        ),
    )


def expiry_seconds(value: str) -> int:
    match = _EXPIRY.fullmatch(value)
    if match is None:
        raise ValueError("Wormkey expiry must use minutes or hours, for example 30m or 1h")
    amount = int(match.group(1))
    multiplier = 60 if match.group(2) == "m" else 60 * 60
    seconds = amount * multiplier
    if not MINIMUM_EXPIRY_SECONDS <= seconds <= MAXIMUM_EXPIRY_SECONDS:
        raise ValueError("Wormkey expiry must be between 5 minutes and 2 hours")
    return seconds


def execute_wormkey_share(plan: WormkeySharePlan) -> int:
    """Run Cadence and Wormkey together until expiry, failure, or operator interruption."""
    configured_secret()
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("npx is required to run the pinned Wormkey package")

    username = "cadence"
    password = secrets.token_urlsafe(24)
    server_environment = os.environ.copy()
    server_environment.update(
        {
            "CADENCE_REVIEW_TUNNEL_BASIC_USERNAME": username,
            "CADENCE_REVIEW_TUNNEL_BASIC_PASSWORD": password,
        }
    )
    tunnel_environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "HOME",
            "LANG",
            "LC_ALL",
            "LOGNAME",
            "PATH",
            "SHELL",
            "TMPDIR",
            "USER",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_STATE_HOME",
        }
    }
    tunnel_environment["npm_config_ignore_scripts"] = "true"
    port = int(plan.local_url.rsplit(":", 1)[1])
    _ensure_port_available(port)
    server_command = (
        sys.executable,
        "-m",
        "cadence.cli",
        "review-serve",
        "--config",
        plan.config_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--external-tunnel",
        "wormkey",
    )
    server: subprocess.Popen[str] | None = None
    tunnel: subprocess.Popen[str] | None = None
    try:
        server = subprocess.Popen(
            server_command,
            env=server_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        _wait_for_server(server, plan.local_url, username, password)
        tunnel_command = (npx, *plan.command[1:])
        tunnel = subprocess.Popen(
            tunnel_command,
            env=tunnel_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if tunnel.stdout is None:
            raise RuntimeError("Wormkey output stream is unavailable")

        ready = False
        for raw_line in tunnel.stdout:
            line = raw_line.strip()
            if not ready and _PUBLIC_URL.fullmatch(line):
                ready = True
                print(
                    json.dumps(
                        {
                            "event": "cadence_review_share_ready",
                            "provider": "wormkey",
                            "url": line,
                            "expires": plan.expires,
                            "basic_auth": {
                                "username": username,
                                "password": password,
                            },
                            "cadence_admin_secret": "provisioned-on-vps-and-not-emitted",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        return_code = tunnel.wait()
        if not ready:
            raise RuntimeError(
                f"Wormkey exited before publishing a share URL (status {return_code})"
            )
        return return_code
    except KeyboardInterrupt:
        return 130
    finally:
        _terminate(tunnel)
        _terminate(server)


def _wait_for_server(
    process: subprocess.Popen[str],
    local_url: str,
    username: str,
    password: str,
    *,
    timeout_seconds: float = 10.0,
) -> None:
    authorization = base64.b64encode(f"{username}:{password}".encode()).decode()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Cadence review server exited during tunnel startup")
        request = Request(
            f"{local_url}/healthz",
            headers={"Authorization": f"Basic {authorization}"},
        )
        try:
            with urlopen(request, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.1)
    raise RuntimeError("Cadence review server did not become ready within 10 seconds")


def _ensure_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"review share port {port} is already in use") from exc


def _terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
