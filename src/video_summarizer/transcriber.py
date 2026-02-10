"""Video transcription: YouTube subtitle extraction, audio download, and whisper transcription."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def _log(msg: str) -> None:
    print(f"[video-summarizer] {msg}", file=sys.stderr)


def _is_youtube_url(source: str) -> bool:
    try:
        parsed = urlparse(source)
        return parsed.hostname in (
            "www.youtube.com",
            "youtube.com",
            "youtu.be",
            "m.youtube.com",
        )
    except Exception:
        return False


def _parse_vtt(vtt_text: str) -> str:
    """Strip VTT timestamps and metadata, return clean text."""
    lines = []
    for line in vtt_text.splitlines():
        line = line.strip()
        # Skip empty, WEBVTT header, timestamps, NOTE lines, style blocks
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if line.startswith("NOTE"):
            continue
        if line.startswith("STYLE"):
            continue
        if "-->" in line:
            continue
        # Skip numeric cue identifiers
        if re.match(r"^\d+$", line):
            continue
        # Strip VTT tags like <c>, </c>, <00:00:01.000>
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)
    # Deduplicate consecutive identical lines (VTT often repeats)
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    return " ".join(deduped)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    _log(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def get_video_info(source: str) -> dict:
    """Get video metadata without transcription."""
    if _is_youtube_url(source):
        result = _run(["yt-dlp", "--dump-json", "--no-download", source])
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp metadata failed: {result.stderr}")
        data = json.loads(result.stdout)
        return {
            "title": data.get("title", "Unknown"),
            "duration_seconds": data.get("duration"),
            "channel": data.get("channel") or data.get("uploader", "Unknown"),
            "upload_date": data.get("upload_date"),
            "view_count": data.get("view_count"),
            "description": (data.get("description") or "")[:500],
            "format": data.get("format_note") or data.get("format", "Unknown"),
        }
    else:
        # Local file — use ffprobe
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        result = _run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ])
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        return {
            "title": Path(source).name,
            "duration_seconds": float(fmt.get("duration", 0)),
            "format": fmt.get("format_long_name", "Unknown"),
            "size_bytes": int(fmt.get("size", 0)),
            "streams": len(data.get("streams", [])),
        }


def _try_youtube_subtitles(source: str, language: str) -> str | None:
    """Try to extract existing YouTube subtitles (instant, no model needed)."""
    with tempfile.TemporaryDirectory(prefix="vs-subs-") as tmpdir:
        out_template = os.path.join(tmpdir, "subs")
        result = _run([
            "yt-dlp",
            "--write-subs", "--write-auto-subs",
            "--sub-lang", language,
            "--sub-format", "vtt",
            "--skip-download",
            "-o", out_template,
            source,
        ])
        if result.returncode != 0:
            _log(f"Subtitle extraction failed: {result.stderr}")
            return None

        # Look for the downloaded subtitle file
        for f in Path(tmpdir).iterdir():
            if f.suffix == ".vtt":
                _log(f"Found subtitle file: {f.name}")
                return _parse_vtt(f.read_text(encoding="utf-8"))
    return None


def _download_audio(source: str) -> tuple[str, dict]:
    """Download audio from YouTube URL. Returns (audio_path, metadata)."""
    tmpdir = tempfile.mkdtemp(prefix="vs-audio-")
    out_path = os.path.join(tmpdir, "audio.%(ext)s")
    result = _run([
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", out_path,
        "--dump-json",
        source,
    ])
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp audio download failed: {result.stderr}")

    # Parse metadata from the JSON output (last JSON object in stdout)
    metadata = {}
    for line in result.stdout.strip().splitlines():
        try:
            metadata = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    # Find the downloaded audio file
    for f in Path(tmpdir).iterdir():
        if f.suffix in (".wav", ".m4a", ".mp3", ".opus", ".webm"):
            return str(f), metadata

    raise RuntimeError(f"No audio file found in {tmpdir}")


def _extract_audio_local(source: str) -> str:
    """Extract audio from local video file using ffmpeg. Returns WAV path."""
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    tmpdir = tempfile.mkdtemp(prefix="vs-local-")
    wav_path = os.path.join(tmpdir, "audio.wav")
    result = _run([
        "ffmpeg", "-i", str(path),
        "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
        "-y", wav_path,
    ])
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    return wav_path


def _whisper_transcribe(audio_path: str, model: str, language: str) -> str:
    """Transcribe audio using mlx-whisper."""
    import mlx_whisper

    _log(f"Transcribing with mlx-whisper model={model} language={language}")
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=f"mlx-community/whisper-{model}-mlx",
        language=language,
    )
    return result.get("text", "").strip()


def _cleanup_path(path: str) -> None:
    """Remove a temp file and its parent directory."""
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
        parent = p.parent
        if parent.exists() and parent.name.startswith("vs-"):
            import shutil
            shutil.rmtree(parent, ignore_errors=True)
    except Exception as e:
        _log(f"Cleanup warning: {e}")


def transcribe_video(
    source: str,
    language: str = "en",
    model: str = "base",
) -> dict:
    """
    Transcribe a video from YouTube URL or local file path.

    Strategy:
    1. YouTube with subtitles → extract VTT (instant)
    2. YouTube without subtitles → download audio → whisper
    3. Local file → ffmpeg extract audio → whisper
    """
    audio_path = None
    try:
        if _is_youtube_url(source):
            # Try subtitles first (instant, free)
            _log("Attempting YouTube subtitle extraction...")
            transcript = _try_youtube_subtitles(source, language)
            if transcript and len(transcript) > 50:
                _log(f"Got subtitles: {len(transcript)} chars")
                info = get_video_info(source)
                return {
                    "transcript": transcript,
                    "metadata": {
                        "method": "youtube_subtitles",
                        "title": info.get("title", "Unknown"),
                        "duration_seconds": info.get("duration_seconds"),
                        "word_count": len(transcript.split()),
                        "language": language,
                    },
                }

            # Fall back to whisper
            _log("No subtitles found, downloading audio for whisper...")
            audio_path, yt_meta = _download_audio(source)
            transcript = _whisper_transcribe(audio_path, model, language)
            return {
                "transcript": transcript,
                "metadata": {
                    "method": "whisper",
                    "model": model,
                    "title": yt_meta.get("title", "Unknown"),
                    "duration_seconds": yt_meta.get("duration"),
                    "word_count": len(transcript.split()),
                    "language": language,
                },
            }
        else:
            # Local file
            _log(f"Extracting audio from local file: {source}")
            audio_path = _extract_audio_local(source)
            transcript = _whisper_transcribe(audio_path, model, language)
            info = get_video_info(source)
            return {
                "transcript": transcript,
                "metadata": {
                    "method": "whisper",
                    "model": model,
                    "title": info.get("title", Path(source).name),
                    "duration_seconds": info.get("duration_seconds"),
                    "word_count": len(transcript.split()),
                    "language": language,
                },
            }
    finally:
        if audio_path:
            _cleanup_path(audio_path)
