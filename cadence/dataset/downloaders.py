"""Replaceable, non-circumventing source inspection and download adapters."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from cadence.dataset.records import SourceRecord


@dataclass(frozen=True)
class SourceInspection:
    supported: bool
    method: str
    title: str | None = None
    publisher_or_creator: str | None = None
    platform: str | None = None
    duration_seconds: float | None = None
    content_length_bytes: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    bytes_written: int
    method: str


class SourceDownloader(Protocol):
    name: str

    def inspect(self, url: str) -> SourceInspection: ...

    def download(self, source: SourceRecord, destination: Path) -> DownloadResult: ...


class DirectHTTPDownloader:
    name = "direct-http"

    def __init__(self, *, timeout_seconds: int = 30, maximum_bytes: int | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.maximum_bytes = maximum_bytes

    def inspect(self, url: str) -> SourceInspection:
        try:
            request = Request(url, method="HEAD", headers={"User-Agent": "Cadence/0.1"})
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get_content_type()
                length_value = response.headers.get("Content-Length")
                content_length = int(length_value) if length_value else None
                video_extension = urlsplit(url).path.lower().endswith(
                    (".mp4", ".mov", ".webm")
                )
                supported = content_type.startswith("video/") or video_extension
                return SourceInspection(
                    supported=supported,
                    method=self.name,
                    title=Path(urlsplit(url).path).name or None,
                    platform=urlsplit(url).netloc.lower(),
                    content_length_bytes=content_length,
                    error=None if supported else f"unsupported content type: {content_type}",
                )
        except Exception as exc:
            return SourceInspection(False, self.name, error=f"inspection failed: {exc}")

    def download(self, source: SourceRecord, destination: Path) -> DownloadResult:
        temporary = destination.with_suffix(destination.suffix + ".part")
        written = 0
        byte_limit = source.content_length_bytes or self.maximum_bytes
        try:
            request = Request(str(source.url), headers={"User-Agent": "Cadence/0.1"})
            response = urlopen(request, timeout=self.timeout_seconds)
            with response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if byte_limit is not None and written > byte_limit:
                        raise ValueError("download exceeded configured maximum reservation")
                    out.write(chunk)
            temporary.replace(destination)
            return DownloadResult(destination, written, self.name)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


class YtDlpDownloader:
    name = "yt-dlp"

    def __init__(self, binary: str = "yt-dlp") -> None:
        self.binary = binary

    def inspect(self, url: str) -> SourceInspection:
        if shutil.which(self.binary) is None:
            return SourceInspection(False, self.name, error="yt-dlp is not installed")
        result = subprocess.run(
            [self.binary, "--dump-single-json", "--skip-download", "--no-playlist", url],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return SourceInspection(
                False, self.name, error=result.stderr.strip() or "unsupported URL"
            )
        metadata = json.loads(result.stdout)
        return SourceInspection(
            supported=True,
            method=self.name,
            title=metadata.get("title"),
            publisher_or_creator=metadata.get("channel") or metadata.get("uploader"),
            platform=metadata.get("extractor_key") or urlsplit(url).netloc,
            duration_seconds=metadata.get("duration"),
            content_length_bytes=metadata.get("filesize") or metadata.get("filesize_approx"),
        )

    def download(self, source: SourceRecord, destination: Path) -> DownloadResult:
        if shutil.which(self.binary) is None:
            raise RuntimeError("yt-dlp is not installed")
        result = subprocess.run(
            [
                self.binary,
                "--no-playlist",
                "--no-part",
                "--restrict-filenames",
                "-f",
                "bestvideo*+bestaudio/best",
                "--merge-output-format",
                "mp4",
                "-o",
                str(destination),
                str(source.url),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not destination.exists():
            destination.unlink(missing_ok=True)
            raise RuntimeError(result.stderr.strip() or "yt-dlp download failed")
        return DownloadResult(destination, destination.stat().st_size, self.name)


class DownloaderChain:
    def __init__(self, adapters: list[SourceDownloader]) -> None:
        self.adapters = adapters

    def inspect(self, url: str) -> tuple[SourceDownloader | None, SourceInspection]:
        errors: list[str] = []
        for adapter in self.adapters:
            inspection = adapter.inspect(url)
            if inspection.supported:
                return adapter, inspection
            if inspection.error:
                errors.append(f"{adapter.name}: {inspection.error}")
        return None, SourceInspection(False, "none", error="; ".join(errors) or "unsupported URL")

    def by_name(self, name: str) -> SourceDownloader:
        for adapter in self.adapters:
            if adapter.name == name:
                return adapter
        raise ValueError(f"download adapter is unavailable: {name}")
