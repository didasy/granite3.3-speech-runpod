````markdown
# Granite Speech Serverless — Specs, Env & API Guide

A tidy, copy-pasteable guide for your Runpod Serverless speech worker (Granite 3.3) with YouTube (cookies via **request**), MP4 via ffmpeg, VAD, SRT, MinIO-only path I/O, size caps, and YouTube pre-download caps.

---

## Table of Contents
- [Specs](#specs)
- [Environments](#environments)
  - [Core / HF / Torch](#core--hf--torch)
  - [Chunking & SRT](#chunking--srt)
  - [VAD](#vad)
  - [YouTube](#youtube)
  - [MinIO (required for `type:"path"` & SRT saves)](#minio-required-for-typepath--srt-saves)
  - [Optional shorthand & defaults](#optional-shorthand--defaults)
- [Requests](#requests)
  - [How to call](#how-to-call)
  - [Request schema](#request-schema)
  - [YouTube options (request-only)](#youtube-options-request-only)
  - [Success response schema](#success-response-schema)
  - [Error response schema](#error-response-schema)
  - [Canonical request examples](#canonical-request-examples)
  - [Notes & tips](#notes--tips)

---

## Specs
| Resource | Minimum |
|---|---|
| **VRAM** | 24 GB |
| **Storage** | 25 GB |

---

## Environments

### Core / HF / Torch
```dotenv
# Model & caches
MODEL_ID=ibm-granite/granite-speech-3.3-8b
HF_HOME=/weights/huggingface
HUGGINGFACE_HUB_CACHE=/weights/huggingface
# HF_HUB_ENABLE_HF_TRANSFER=1      # Optional fast downloads (requires `pip install hf_transfer`)
# If not installing hf_transfer, set HF_HUB_ENABLE_HF_TRANSFER=0 or omit it.

# Runtime
TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
````

> ℹ️ `TRANSFORMERS_CACHE` is deprecated by 🤗 Transformers v5; prefer `HF_HOME`/`HUGGINGFACE_HUB_CACHE`.

### Chunking & SRT

```dotenv
# Fixed-window chunking (used when VAD is off, or as a fallback)
CHUNK_SECONDS=20
CHUNK_OVERLAP=1

# Input length caps (minutes)
# Example: MAX_INPUT_MINUTES=10  (0 disables minutes cap)
# Action when over cap: reject (default) or truncate
MAX_INPUT_MINUTES=0
MAX_INPUT_ACTION=reject

# Legacy seconds cap (still supported)
# The strictest cap is enforced among:
#   MAX_DURATION_SECONDS, (MAX_INPUT_MINUTES * 60), and per-request max_input_minutes
MAX_DURATION_SECONDS=0

# SRT formatting
SRT_MAX_CHARS_PER_LINE=42
SRT_MAX_LINES=2
RETURN_SRT_BASE64_DEFAULT=true
```

> ℹ️ **Cap precedence:** The handler enforces the *strictest* of `MAX_DURATION_SECONDS`, `MAX_INPUT_MINUTES * 60`, and the per-request `max_input_minutes`.

### VAD

```dotenv
USE_VAD_DEFAULT=false
VAD_AGGRESSIVENESS=2        # 0..3
VAD_FRAME_MS=30             # 10 | 20 | 30
VAD_MIN_SPEECH_MS=300
VAD_MAX_SILENCE_MS=400
VAD_PAD_MS=120
VAD_MAX_SEGMENT_SECONDS=30
```

### YouTube

```dotenv
# Pre-download cap in minutes (0 disables pre-download cap)
# Example: MAX_YOUTUBE_MINUTES=10
MAX_YOUTUBE_MINUTES=0

YTDLP_FORMAT=bestaudio/best
YTDLP_RETRIES=3
```

> ✅ **Cookies, proxy, and player options are request-only** (see below). No env needed for those now.

### MinIO (required for `type:"path"` and to save SRTs)

```dotenv
MINIO_ENDPOINT=minio.yourdomain:9000
MINIO_SECURE=true          # set false if using HTTP
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
```

### (optional shorthand & defaults)

```dotenv
MINIO_INPUT_BUCKET=media
MINIO_INPUT_PREFIX=in/
MINIO_OUTPUT_BUCKET=media
MINIO_OUTPUT_PREFIX=out/
```

---

## Requests

### Granite Speech Serverless — Request & Response Guide

This section covers **all supported request shapes** and **response formats** for your handler.

---

### How to call

Send a `POST` to your Runpod worker with a body shaped like:

```jsonc
{
  "input": {
    // ...payload fields described below...
  }
}
```

> All examples below show only the `input` object for brevity. When you call the worker, wrap it as `{"input": { ... }}`.

---

### Request schema

**Top-level fields (all optional unless marked “required”):**

* **`audio`** *(required)* — one of:

  * **String (URL)**

    * `https://...` (mp3, wav, m4a, mp4, etc.)
    * YouTube URL (`https://www.youtube.com/...` or `https://youtu.be/...`)
  * **String (Data URL)**

    * `data:audio/<mime>;base64,<...>`
  * **String (MinIO path)**

    * `minio://bucket/key`, `s3://bucket/key`, or bare `bucket/key` (MinIO-only; **local paths are rejected**)
  * **Object**

    * `{ "type": "url", "value": "https://..." }`
    * `{ "type": "data_url", "value": "data:audio/...;base64,..." }`
    * `{ "type": "base64", "value": "<BASE64_BYTES>" }`
    * `{ "type": "path", "value": "minio://bucket/key" }`
* **`task`** — `"transcribe"` (default) or `"translate"`
* **`prompt`** — custom instruction text (the handler auto-prepends `<|audio|>` if you don’t)
* **`source_lang`** — for translate mode (e.g., `"es"`, `"auto"`)
* **`target_lang`** — for translate mode (default `"en"`)
* **`use_vad`** — boolean; enables WebRTC VAD segmentation (default from `USE_VAD_DEFAULT=false`)
* **`make_srt`** — boolean; return/generate SRT blocks (default `false`)
* **`srt_path`** — **MinIO path only** to store `.srt` (e.g., `minio://media/out/file.srt`, `media/out/file.srt`)

  * If omitted and `make_srt=true`, the handler auto-writes to `MINIO_OUTPUT_BUCKET/PREFIX/` if configured.
* **`return_srt_base64`** — boolean; include `srt_base64` in response (default from `RETURN_SRT_BASE64_DEFAULT=true`)
* **`max_new_tokens`** — integer; default `256`
* **`temperature`** — float; default `0.0` (greedy). >0 enables sampling.
* **`num_beams`** — integer; default `1`

**Chunking / VAD detail knobs (optional):**

* **`max_input_minutes`** — float; per-request cap (interacts with env caps)
* **`max_input_action`** — `"reject"` or `"truncate"`; default from env
* **`max_youtube_minutes`** — float; pre-download cap for YouTube (see env)
* **Advanced (usually leave to env defaults):**
  `CHUNK_SECONDS`, `CHUNK_OVERLAP`,
  `VAD_AGGRESSIVENESS` (0..3), `VAD_FRAME_MS` (10|20|30),
  `VAD_MIN_SPEECH_MS`, `VAD_MAX_SILENCE_MS`, `VAD_PAD_MS`, `VAD_MAX_SEGMENT_SECONDS`

**Important rules:**

* ✅ Local filesystem **is disabled**. Any `"type":"path"` **must** be MinIO: `minio://bucket/key`, `s3://bucket/key`, or `bucket/key`.
* ✅ `srt_path` must be MinIO (same formats). Local paths are rejected.
* ✅ YouTube downloads honor the strictest applicable minutes cap *before* fetching audio.

---

### YouTube options (request-only)

These are **per-request** and override any defaults:

* `yt_cookies_b64` — base64 of a Netscape `cookies.txt`
* `yt_cookies_text` — raw Netscape cookie text inline
* `yt_cookie_path` — MinIO path to `cookies.txt` (e.g., `minio://secrets/yt/cookies.txt`)
* `yt_proxy` — e.g., `http://user:pass@host:8080`
* `yt_geo_country` — e.g., `"US"`
* `yt_user_agent` — UA string
* `yt_accept_language` — e.g., `"en-US,en;q=0.8"`
* `yt_player_clients` — list, e.g., `["android","web"]`
* `yt_no_check_cert` — boolean
* `yt_format` — yt-dlp format string (defaults to env `YTDLP_FORMAT`)

> Use cookies/proxy when encountering **HTTP 403 Forbidden** from YouTube.

---

### Success response schema

```jsonc
{
  "status": "ok",
  "text": "<full transcript or translation text>",
  "srt": "<SRT content or null>",
  "srt_base64": "<base64-encoded SRT or null>",
  "srt_path_written": "minio://bucket/key or null",
  "timings": {
    "load_seconds": 0.0,
    "inference_seconds": 0.0
  },
  "meta": {
    "device": "cuda:0 | cpu",
    "dtype": "torch.bfloat16 | float16 | float32",
    "sr_hz": 16000,
    "task": "transcribe | translate",
    "model_id": "ibm-granite/granite-speech-3.3-8b",
    "audio_seconds": 0.0,
    "chunks": 0,                        // present when VAD or chunking used
    "use_vad": false,
    "download_caps_sec": {              // present for YouTube path
      "youtube_predownload": null
    },
    "vad": {                            // present when VAD/chunking path was used
      "aggr": 2,
      "frame_ms": 30,
      "min_speech_ms": 300,
      "max_silence_ms": 400,
      "pad_ms": 120,
      "max_segment_s": 30.0
    },
    "tokens": {
      "prompt_tokens": 0,               // fast path
      "new_tokens": 0,                  // fast path
      "new_tokens_sum": 0               // VAD/chunked path
    }
  }
}
```

---

### Error response schema

```jsonc
{
  "status": "error",
  "error": "<human-readable message>"
}
```

**Typical reasons**

* Missing `"audio"`
* Local file path used (local I/O disabled)
* SRT requested but MinIO not configured
* Input length exceeds cap and `max_input_action="reject"`
* MinIO credentials not configured (for `"type":"path"` or `srt_path`)
* Unsupported input type
* YouTube 403 (supply cookies/proxy/geo; see request-only options)

---

### Canonical request examples

**1) Basic URL → transcript only**

```json
{
  "task": "transcribe",
  "audio": { "type": "url", "value": "https://example.com/clip.mp3" }
}
```

**2) YouTube → VAD + SRT (auto-saved), pre-download cap 10 min**

```json
{
  "task": "transcribe",
  "audio": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "use_vad": true,
  "make_srt": true,
  "max_youtube_minutes": 10
}
```

**3) MinIO path in → MinIO path out**

```json
{
  "task": "transcribe",
  "audio": { "type": "path", "value": "minio://media/in/meeting.mp4" },
  "use_vad": true,
  "make_srt": true,
  "srt_path": "minio://media/out/meeting.srt"
}
```

**4) Shorthand MinIO path (bucket/key) + auto SRT to default bucket/prefix**

```json
{
  "task": "transcribe",
  "audio": { "type": "path", "value": "media/in/townhall.mp3" },
  "make_srt": true
}
```

**5) Base64 payload**

```json
{
  "task": "transcribe",
  "audio": { "type": "base64", "value": "<BASE64_AUDIO_BYTES>" }
}
```

**6) Data URL payload**

```json
{
  "task": "transcribe",
  "audio": "data:audio/wav;base64,AAA..."
}
```

**7) Translation (ES → EN) with light sampling**

```json
{
  "task": "translate",
  "source_lang": "es",
  "target_lang": "en",
  "temperature": 0.2,
  "audio": "https://example.com/spanish-talk.wav",
  "make_srt": true
}
```

**8) Fixed windowing (VAD off) with custom chunking**

```json
{
  "task": "transcribe",
  "use_vad": false,
  "audio": "https://example.com/lecture.m4a",
  "make_srt": true,
  "CHUNK_SECONDS": 30,
  "CHUNK_OVERLAP": 2
}
```

**9) Length caps — reject if over 12 minutes**

```json
{
  "task": "transcribe",
  "audio": "https://example.com/long.mp4",
  "max_input_minutes": 12,
  "max_input_action": "reject"
}
```

**10) Length caps — truncate to 8 minutes**

```json
{
  "task": "transcribe",
  "audio": { "type": "path", "value": "minio://media/in/meeting.mp4" },
  "max_input_minutes": 8,
  "max_input_action": "truncate",
  "make_srt": true,
  "srt_path": "minio://media/out/meeting_8min.srt"
}
```

**11) YouTube with cookies (base64) to bypass 403**

```json
{
  "task": "transcribe",
  "audio": "https://www.youtube.com/watch?v=fIM0Cxn8uNc",
  "make_srt": true,
  "max_youtube_minutes": 10,
  "yt_cookies_b64": "<BASE64_OF_NETSCAPE_COOKIES_TXT>"
}
```

**12) YouTube with MinIO cookies + geo**

```json
{
  "task": "transcribe",
  "audio": "https://www.youtube.com/watch?v=fIM0Cxn8uNc",
  "make_srt": true,
  "yt_cookie_path": "minio://secrets/yt/cookies.txt",
  "yt_geo_country": "US"
}
```

**13) YouTube via proxy + custom players**

```json
{
  "task": "transcribe",
  "audio": "https://www.youtube.com/watch?v=fIM0Cxn8uNc",
  "make_srt": true,
  "yt_proxy": "http://user:pass@proxyhost:8080",
  "yt_player_clients": ["android","web"]
}
```

---

## Notes & tips

* Prefer `use_vad=true` for conversational audio; turn it off for music or extremely noisy content.
* For very long inputs, use `max_input_minutes` with `"truncate"` to bound cost and latency.
* When using YouTube, add `max_youtube_minutes` to prevent oversized downloads *and* still respect post-load caps.
* If enabling `HF_HUB_ENABLE_HF_TRANSFER=1`, remember to `pip install hf_transfer` (or set it to `0`/omit).

```
::contentReference[oaicite:0]{index=0}
```
