# Video Summarizer MCP Server

An MCP server that transcribes videos so Claude Code can summarize them — no separate LLM API costs. Works with YouTube URLs and local video files.

## How It Works

```
You: "Summarize this video: https://youtube.com/watch?v=..."
         ↓
Claude Code → calls transcribe_video tool
         ↓
MCP Server → extracts transcript (subtitles or whisper)
         ↓
Claude Code ← reads transcript and summarizes
```

## Tools

### `transcribe_video_tool`

Transcribes a video from YouTube or a local file.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | string | required | YouTube URL or absolute local file path |
| `language` | string | `"en"` | ISO 639-1 language code |
| `model` | string | `"base"` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |

**Transcription strategy (smart fallback):**
1. YouTube with subtitles → extracts VTT subs (instant, no model needed)
2. YouTube without subtitles → downloads audio → mlx-whisper
3. Local video file → ffmpeg extracts audio → mlx-whisper

### `get_video_info_tool`

Returns video metadata without transcription (fast).

| Param | Type | Description |
|-------|------|-------------|
| `source` | string | YouTube URL or absolute local file path |

Returns title, duration, channel, format, view count.

## Setup

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)
- Apple Silicon Mac (for mlx-whisper acceleration)

### Install

```bash
git clone https://github.com/samyakgoyal/video-summarizer.git
cd video-summarizer
uv sync
```

### Register with Claude Code

Add to your `~/.mcp.json`:

```json
{
  "mcpServers": {
    "video-summarizer": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/video-summarizer",
        "python",
        "-m",
        "video_summarizer"
      ]
    }
  }
}
```

Restart Claude Code to pick up the new server.

## Usage Examples

Once registered, just ask Claude Code naturally:

- "Summarize this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
- "What are the key points in /Users/me/Downloads/lecture.mp4?"
- "Get info about this video: https://youtu.be/jNQXAC9IVRw"

For longer videos or non-English content, specify the model and language:

- "Transcribe this Japanese video with the large model: https://youtube.com/watch?v=..."

## Tech Stack

- **[mcp](https://pypi.org/project/mcp/)** — MCP Python SDK (stdio transport)
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — YouTube subtitle/audio extraction
- **[mlx-whisper](https://github.com/ml-explore/mlx-examples)** — Apple Silicon optimized whisper (no PyTorch dependency)
- **ffmpeg** — Local video audio extraction
