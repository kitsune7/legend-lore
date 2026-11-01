"""Step 1 (URL variant) — stream YouTube audio straight into a 16 kHz mono WAV.

yt-dlp writes the raw media stream to its stdout, ffmpeg reads it from stdin and
writes the WAV incrementally, so no full-size source file (webm/m4a) is stored.
"""
from __future__ import annotations

import collections
import re
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_PREFIXES = {"shorts", "live", "embed"}

# Workaround for YouTube's SABR streaming block (the default web client fails).
# Same override as scripts/fetch_clip.sh; revisit on yt-dlp upgrades.
PLAYER_CLIENTS = ("ios", "android", "tv")

_STDERR_TAIL_BYTES = 2048


class YoutubeError(RuntimeError):
    """Video unavailable, or the download/transcode pipe failed."""


def _parse(s: str):
    p = urlparse(s.strip())
    if p.scheme not in ("http", "https") or (p.hostname or "").lower() not in _HOSTS:
        return None
    return p


def is_youtube_host(s: str) -> bool:
    """True for any http(s) URL on a YouTube host, even if it has no single-video ID."""
    return _parse(s) is not None


def video_id(s: str) -> str | None:
    """Return the 11-char video ID for a supported single-video URL, else None."""
    p = _parse(s)
    if p is None:
        return None
    host = (p.hostname or "").lower()
    parts = [x for x in p.path.split("/") if x]
    if host == "youtu.be":
        cand = parts[0] if parts else None
    elif parts and parts[0] == "watch":
        cand = (parse_qs(p.query).get("v") or [None])[0]
    elif len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
        cand = parts[1]
    else:
        cand = None
    return cand if cand and _ID_RE.match(cand) else None


def check_available(url: str) -> dict:
    """Metadata-only lookup; raise YoutubeError for live/upcoming/unavailable videos."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": list(PLAYER_CLIENTS)}},
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as e:
        raise YoutubeError(f"cannot fetch {url}: {e}") from e

    status = info.get("live_status")
    if status in ("is_live", "is_upcoming"):
        raise YoutubeError(
            f"{url} is {status.replace('is_', '').replace('_', ' ')}; "
            "only finished videos (including past live VODs) are supported"
        )
    if not info.get("formats"):
        raise YoutubeError(f"{url} has no downloadable formats")
    return info


def _drain(stream, tail: collections.deque, echo: bool) -> None:
    """Read `stream` to EOF, keeping a bounded byte tail and optionally echoing it."""
    while True:
        chunk = stream.read(4096)
        if not chunk:
            return
        tail.extend(chunk)
        if echo:
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()


def _tail(buf: collections.deque) -> str:
    return bytes(buf).decode("utf-8", "replace").strip()


def stream_to_16k_mono(url: str, dst: str | Path) -> Path:
    """Stream `url`'s audio through ffmpeg into a 16 kHz mono WAV at `dst`.

    Any existing `dst` is overwritten. A partial `dst` is removed on failure.
    """
    check_available(url)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ytdlp_cmd = [
        sys.executable, "-m", "yt_dlp", "--no-playlist",
        "--extractor-args", "youtube:player_client=" + ",".join(PLAYER_CLIENTS),
        "-f", "bestaudio/best", "-o", "-", url,
    ]
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0",
        "-vn", "-ac", "1", "-ar", "16000", str(dst),
    ]

    ytdlp_err: collections.deque = collections.deque(maxlen=_STDERR_TAIL_BYTES)
    ffmpeg_err: collections.deque = collections.deque(maxlen=_STDERR_TAIL_BYTES)

    ytdlp = ffmpeg = None
    threads: list[threading.Thread] = []
    try:
        ytdlp = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=ytdlp.stdout, stderr=subprocess.PIPE, bufsize=0)
        # Drop our copy so ffmpeg exiting early delivers SIGPIPE to yt-dlp.
        assert ytdlp.stdout is not None
        ytdlp.stdout.close()

        for proc, buf, echo in ((ytdlp, ytdlp_err, True), (ffmpeg, ffmpeg_err, False)):
            t = threading.Thread(target=_drain, args=(proc.stderr, buf, echo), daemon=True)
            t.start()
            threads.append(t)

        ffmpeg_rc = ffmpeg.wait()
        ytdlp_rc = ytdlp.wait()
        for t in threads:
            t.join()
    except BaseException:
        try:
            for proc in (ytdlp, ffmpeg):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
            for proc in (ytdlp, ffmpeg):
                if proc is not None:
                    proc.wait()
        finally:
            dst.unlink(missing_ok=True)
        raise

    failures = []
    if ytdlp_rc != 0:
        failures.append(f"yt-dlp exited {ytdlp_rc}:\n{_tail(ytdlp_err)}")
    if ffmpeg_rc != 0:
        failures.append(f"ffmpeg exited {ffmpeg_rc}:\n{_tail(ffmpeg_err)}")
    if failures:
        dst.unlink(missing_ok=True)
        raise YoutubeError(f"streaming {url} failed\n" + "\n".join(failures))
    return dst
