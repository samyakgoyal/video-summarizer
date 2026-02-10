"""MCP server for video transcription."""

import json
import sys
import traceback

from mcp.server.fastmcp import FastMCP

from video_summarizer.transcriber import get_video_info, transcribe_video

mcp = FastMCP("video-summarizer")

VALID_MODELS = ("tiny", "base", "small", "medium", "large")


@mcp.tool()
def transcribe_video_tool(
    source: str,
    language: str = "en",
    model: str = "base",
) -> str:
    """Transcribe a video from a YouTube URL or local file path.

    For YouTube: tries subtitle extraction first (instant), falls back to whisper.
    For local files: extracts audio with ffmpeg, transcribes with whisper.

    Args:
        source: YouTube URL (youtube.com/youtu.be) or absolute local file path
        language: ISO 639-1 language code (default: "en")
        model: Whisper model size: tiny, base, small, medium, large (default: "base")
    """
    if model not in VALID_MODELS:
        return json.dumps({"error": f"Invalid model '{model}'. Choose from: {', '.join(VALID_MODELS)}"})
    try:
        result = transcribe_video(source, language=language, model=model)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_video_info_tool(source: str) -> str:
    """Get metadata about a video without transcribing it.

    Returns title, duration, channel/format info. Fast — no transcription.

    Args:
        source: YouTube URL or absolute local file path
    """
    try:
        result = get_video_info(source)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
