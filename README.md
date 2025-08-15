# Specs

- 24GB VRAM
- 25GB Storage

# Enviroments

## Core / HF / Torch

MODEL_ID=ibm-granite/granite-speech-3.3-8b

HF_HOME=/weights/huggingface

HUGGINGFACE_HUB_CACHE=/weights/huggingface

TRANSFORMERS_CACHE=/weights/huggingface

HF_HUB_ENABLE_HF_TRANSFER=1

TOKENIZERS_PARALLELISM=false

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

## Chunking / SRT

CHUNK_SECONDS=20

CHUNK_OVERLAP=1

MAX_INPUT_MINUTES — e.g. 10 (0 disables minutes cap)

MAX_INPUT_ACTION — reject (default) or truncate

(still supported) MAX_DURATION_SECONDS — legacy seconds cap; we enforce the strictest among MAX_DURATION_SECONDS, MAX_INPUT_MINUTES*60, and per-request max_input_minutes.

SRT_MAX_CHARS_PER_LINE=42

SRT_MAX_LINES=2

RETURN_SRT_BASE64_DEFAULT=true

## VAD

USE_VAD_DEFAULT=false

VAD_AGGRESSIVENESS=2

VAD_FRAME_MS=30

VAD_MIN_SPEECH_MS=300

VAD_MAX_SILENCE_MS=400

VAD_PAD_MS=120

VAD_MAX_SEGMENT_SECONDS=30

## YouTube

MAX_YOUTUBE_MINUTES — e.g. 10 (0 disables pre-download YT cap)

YTDLP_FORMAT=bestaudio/best

YTDLP_RETRIES=3

## MinIO (required to use "type":"path" and to save SRTs)

MINIO_ENDPOINT=minio.yourdomain:9000

MINIO_SECURE=true (false if HTTP)

MINIO_ACCESS_KEY=...

MINIO_SECRET_KEY=...

## (optional shorthand & defaults)

MINIO_INPUT_BUCKET=media

MINIO_INPUT_PREFIX=in/

MINIO_OUTPUT_BUCKET=media

MINIO_OUTPUT_PREFIX=out/

# Requests

# Granite Speech Serverless — Request & Response Guide

This doc captures **all supported request shapes** and **response formats** for the Runpod Serverless handler you deployed (with YouTube, MP4 via ffmpeg, VAD, SRT, MinIO-only path I/O, size caps, and YouTube pre-download cap).

---

## How to call

Send a POST to your Runpod worker with a JSON body shaped like:

    {
      "input": {
        ...payload fields described below...
      }
    }

> All examples below show only the `input` object for brevity. When you call the worker, wrap it as `{"input": { ... }}`.

---

## Request schema

**Top-level fields (all optional unless marked “required”):**

- **`audio`** _(required)_: one of the following forms  
  - **String (URL)**:  
    - `https://...` (mp3, wav, m4a, mp4, etc.)  
    - YouTube URL (`https://www.youtube.com/...` or `https://youtu.be/...`)
  - **String (Data URL)**:  
    - `data:audio/<mime>;base64,<...>`
  - **String (MinIO path)**:  
    - `minio://bucket/key` or `s3://bucket/key` or bare `bucket/key` (MinIO-only; local paths are rejected)
  - **Object**:  
    - `{ "type": "url", "value": "https://..." }`  
    - `{ "type": "data_url", "value": "data:audio/...;base64,..." }`  
    - `{ "type": "base64", "value": "<BASE64_BYTES>" }`  
    - `{ "type": "path", "value": "minio://bucket/key" }`
- **`task`**: `"transcribe"` (default) or `"translate"`
- **`prompt`**: custom instruction text (the handler prepends `<|audio|>` automatically if you don’t)
- **`source_lang`**: language code for translate mode (e.g., `"es"`, `"auto"`)  
- **`target_lang`**: language code for translate mode (default `"en"`)
- **`use_vad`**: boolean, enables WebRTC VAD smart segmentation (default from env `USE_VAD_DEFAULT=false`)
- **`make_srt`**: boolean, return/generate SRT blocks (default `false`)
- **`srt_path`**: **MinIO path only** to store `.srt` (e.g., `minio://media/out/file.srt`, `media/out/file.srt`)  
  - If omitted and `make_srt=true`, the handler auto-writes to `MINIO_OUTPUT_BUCKET/PREFIX/` if configured.
- **`return_srt_base64`**: boolean, include `srt_base64` in response (default from env `RETURN_SRT_BASE64_DEFAULT=true`)
- **`max_new_tokens`**: integer, default `256`
- **`temperature`**: float, default `0.0` (greedy). >0 enables sampling.
- **`num_beams`**: integer, default `1`

**Chunking / VAD detail knobs (optional):**

- **`max_input_minutes`**: float minutes; per-request cap (see also env `MAX_INPUT_MINUTES`)
- **`max_input_action`**: `"reject"` or `"truncate"`; default from env `MAX_INPUT_ACTION`
- **`max_youtube_minutes`**: float minutes; pre-download cap for YouTube (see env `MAX_YOUTUBE_MINUTES`)
- **Advanced (usually leave to env defaults):**
  - `CHUNK_SECONDS` (float), `CHUNK_OVERLAP` (float)
  - `VAD_AGGRESSIVENESS` (0..3), `VAD_FRAME_MS` (10|20|30),  
    `VAD_MIN_SPEECH_MS`, `VAD_MAX_SILENCE_MS`, `VAD_PAD_MS`, `VAD_MAX_SEGMENT_SECONDS`

**Important rules:**

- Local filesystem **is disabled**. Any `"type":"path"` **must** be MinIO: `minio://bucket/key`, `s3://bucket/key`, or `bucket/key`.
- `srt_path` must be MinIO (same formats). Local paths are rejected.
- YouTube downloads honor the strictest applicable minutes cap *before* fetching audio.

---

## Success response schema

    {
      "status": "ok",
      "text": "<full transcript or translation text>",
      "srt": "<SRT content or null>",
      "srt_base64": "<base64-encoded SRT or null>",
      "srt_path_written": "minio://bucket/key or null",
      "timings": {
        "load_seconds": <float>,
        "inference_seconds": <float>
      },
      "meta": {
        "device": "cuda:0 or cpu",
        "dtype": "torch.bfloat16|float16|float32",
        "sr_hz": 16000,
        "task": "transcribe|translate",
        "model_id": "ibm-granite/granite-speech-3.3-8b",
        "audio_seconds": <float>,
        "chunks": <int>,                 // when VAD or chunking was used
        "use_vad": <bool>,
        "download_caps_sec": {           // present when YouTube path was taken
          "youtube_predownload": <float|null>
        },
        "vad": {                         // present when VAD/chunking path was used
          "aggr": <int>,
          "frame_ms": <int>,
          "min_speech_ms": <int>,
          "max_silence_ms": <int>,
          "pad_ms": <int>,
          "max_segment_s": <float>
        },
        "tokens": {
          "prompt_tokens": <int>,        // fast path
          "new_tokens": <int>,           // fast path
          "new_tokens_sum": <int>        // VAD/chunked path
        }
      }
    }

---

## Error response schema

    {
      "status": "error",
      "error": "<human-readable message>"
    }

Typical reasons:
- Missing `"audio"`
- Local file path used (local I/O disabled)
- SRT requested but MinIO not configured
- Input length exceeds cap and `max_input_action="reject"`
- MinIO credentials not configured (for `"type":"path"` or `srt_path`)
- Unsupported input type

---

## Canonical request examples

### 1) Basic URL → transcript only
    {
      "task": "transcribe",
      "audio": { "type": "url", "value": "https://example.com/clip.mp3" }
    }

### 2) YouTube → VAD + SRT (auto-saved), pre-download cap 10 min
    {
      "task": "transcribe",
      "audio": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      "use_vad": true,
      "make_srt": true,
      "max_youtube_minutes": 10
    }

### 3) MinIO path in → MinIO path out
    {
      "task": "transcribe",
      "audio": { "type": "path", "value": "minio://media/in/meeting.mp4" },
      "use_vad": true,
      "make_srt": true,
      "srt_path": "minio://media/out/meeting.srt"
    }

### 4) Shorthand MinIO path (bucket/key) + auto SRT to default bucket/prefix
    {
      "task": "transcribe",
      "audio": { "type": "path", "value": "media/in/townhall.mp3" },
      "make_srt": true
    }

### 5) Base64 payload
    {
      "task": "transcribe",
      "audio": { "type": "base64", "value": "<BASE64_AUDIO_BYTES>" }
    }

### 6) Data URL payload
    {
      "task": "transcribe",
      "audio": "data:audio/wav;base64,AAA..."
    }

### 7) Translation (ES → EN) with light sampling
    {
      "task": "translate",
      "source_lang": "es",
      "target_lang": "en",
      "temperature": 0.2,
      "audio": "https://example.com/spanish-talk.wav",
      "make_srt": true
    }

### 8) Fixed windowing (VAD off) with custom chunking
    {
      "task": "transcribe",
      "use_vad": false,
      "audio": "https://example.com/lecture.m4a",
      "make_srt": true,
      "CHUNK_SECONDS": 30,
      "CHUNK_OVERLAP": 2
    }

### 9) Length caps — reject if over 12 minutes
    {
      "task": "transcribe",
      "audio": "https://example.com/long.mp4",
      "max_input_minutes": 12,
      "max_input_action": "reject"
    }

### 10) Length caps — truncate to 8 minutes
    {
      "task": "transcribe",
      "audio": { "type": "path", "value": "minio://media/in/meeting.mp4" },
      "max_input_minutes": 8,
      "max_input_action": "truncate",
      "make_srt": true,
      "srt_path": "minio://media/out/meeting_8min.srt"
    }

---

## Representative success response (truncated)

    {
      "status": "ok",
      "text": "Hello everyone and welcome ...",
      "srt": "1\n00:00:00,000 --> 00:00:05,120\nHello everyone...\n\n2\n00:00:05,120 --> 00:00:09,900\nToday we will...\n",
      "srt_base64": "MS0uLg==",
      "srt_path_written": "minio://media/out/meeting.srt",
      "timings": { "load_seconds": 5.231, "inference_seconds": 18.442 },
      "meta": {
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "sr_hz": 16000,
        "task": "transcribe",
        "model_id": "ibm-granite/granite-speech-3.3-8b",
        "audio_seconds": 479.812,
        "chunks": 22,
        "use_vad": true,
        "download_caps_sec": { "youtube_predownload": 600.0 },
        "vad": {
          "aggr": 2,
          "frame_ms": 30,
          "min_speech_ms": 300,
          "max_silence_ms": 400,
          "pad_ms": 120,
          "max_segment_s": 30.0
        },
        "tokens": { "new_tokens_sum": 3421 }
      }
    }

---

## Representative error responses

### A) Missing audio
    {
      "status": "error",
      "error": "Missing 'audio' in input."
    }

### B) Local file path blocked
    {
      "status": "error",
      "error": "Local file access is disabled. Use MinIO paths like 'minio://bucket/key' or 'bucket/key'."
    }

### C) SRT requested but MinIO not configured
    {
      "status": "error",
      "error": "SRT requested but no MINIO_OUTPUT_BUCKET configured."
    }

### D) Exceeded input length (reject)
    {
      "status": "error",
      "error": "Input audio is 15.20 min, exceeds limit of 12.00 min. Set 'max_input_action':'truncate' to auto-trim, or lower 'max_input_minutes'."
    }

### E) MinIO not configured (using path or srt_path)
    {
      "status": "error",
      "error": "MinIO not configured: set MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY."
    }

---

## Notes & tips

- Prefer `use_vad=true` for conversational audio; turn it off for music or extremely noisy content.  
- For very long inputs, use `max_input_minutes` with `"truncate"` to bound cost and latency.  
- When using YouTube, add `max_youtube_minutes` to prevent oversized downloads *and* still respect post-load caps.

---
