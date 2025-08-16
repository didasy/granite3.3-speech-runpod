import os, re, io, time, json, shutil, tempfile, subprocess, base64
from typing import Dict, Any, Tuple, Optional, List
from urllib.parse import urlparse

import torch
import torchaudio
import soundfile as sf
import requests
import runpod
import webrtcvad
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from minio import Minio

# ============================================================
# Environment / config
# ============================================================
MODEL_ID = os.getenv("MODEL_ID", "ibm-granite/granite-speech-3.3-8b")

# Audio normalization
SR_TARGET = 16000
MONO_CH = 1

# Fixed-window chunking (when VAD is off)
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "20"))
CHUNK_OVERLAP = float(os.getenv("CHUNK_OVERLAP", "1"))
# Legacy hard seconds cap (0 = off)
MAX_DURATION_SECONDS = float(os.getenv("MAX_DURATION_SECONDS", "0"))

# Global cap in minutes (+ action)
MAX_INPUT_MINUTES = float(os.getenv("MAX_INPUT_MINUTES", "0"))  # 0 = off
MAX_INPUT_ACTION = os.getenv("MAX_INPUT_ACTION", "reject").lower()  # "reject" | "truncate"

# YouTube pre-download cap (minutes)
MAX_YOUTUBE_MINUTES = float(os.getenv("MAX_YOUTUBE_MINUTES", "0"))  # 0 = off

# SRT formatting
SRT_MAX_CHARS_PER_LINE = int(os.getenv("SRT_MAX_CHARS_PER_LINE", "42"))
SRT_MAX_LINES = int(os.getenv("SRT_MAX_LINES", "2"))
RETURN_SRT_BASE64_DEFAULT = os.getenv("RETURN_SRT_BASE64_DEFAULT", "true").lower() == "true"

# YouTube base opts
YTDLP_FORMAT = os.getenv("YTDLP_FORMAT", "bestaudio/best")
YTDLP_RETRIES = int(os.getenv("YTDLP_RETRIES", "3"))

# MinIO (S3 compatible)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")    # e.g. "s3.us-east-005.backblazeb2.com"
MINIO_SECURE = os.getenv("MINIO_SECURE", "true").lower() == "true"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

# Optional shorthand defaults
MINIO_INPUT_BUCKET = os.getenv("MINIO_INPUT_BUCKET")
MINIO_INPUT_PREFIX = os.getenv("MINIO_INPUT_PREFIX", "")
MINIO_OUTPUT_BUCKET = os.getenv("MINIO_OUTPUT_BUCKET")
MINIO_OUTPUT_PREFIX = os.getenv("MINIO_OUTPUT_PREFIX", "out/")

# Disable local user file access (temp files for yt/ffmpeg are still used)
DISABLE_LOCAL_IO = True

# VAD controls
USE_VAD_DEFAULT = os.getenv("USE_VAD_DEFAULT", "false").lower() == "true"
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))  # 0..3
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", "30"))             # 10|20|30
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "300"))
VAD_MAX_SILENCE_MS = int(os.getenv("VAD_MAX_SILENCE_MS", "400"))
VAD_PAD_MS = int(os.getenv("VAD_PAD_MS", "120"))
VAD_MAX_SEGMENT_SECONDS = float(os.getenv("VAD_MAX_SEGMENT_SECONDS", "30"))

# hf_transfer fail-safe
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").lower() in ("1", "true", "yes"):
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ============================================================
# Device / dtype
# ============================================================
if torch.cuda.is_available():
    major_cc, _ = torch.cuda.get_device_capability(0)
    TORCH_DTYPE = torch.bfloat16 if major_cc >= 8 else torch.float16
else:
    TORCH_DTYPE = torch.float32

# ============================================================
# Model load (cold start)
# ============================================================
LOAD_T0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = getattr(processor, "tokenizer", None)
if tokenizer is None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=TORCH_DTYPE
)
model.eval()
MODEL_DEVICE = model.get_input_embeddings().weight.device
LOAD_T1 = time.time()

# ============================================================
# Helpers: YouTube & ffmpeg
# ============================================================
YOUTUBE_RX = re.compile(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+", re.IGNORECASE)
def _is_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_RX.match(url or ""))

def _run_ffmpeg_to_wav(in_path: str, out_path: str, sr: int = SR_TARGET) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path, "-vn", "-ac", str(MONO_CH), "-ar", str(sr), "-f", "wav", out_path
    ]
    subprocess.run(cmd, check=True)

def _bytes_to_tempfile(raw: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path

def _read_wav_bytes(raw_wav: bytes) -> Tuple[torch.Tensor, int]:
    data, sr = sf.read(io.BytesIO(raw_wav), always_2d=False)
    if data.ndim == 2:
        data = data.T
    wav = torch.tensor(data, dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav, sr

def _decode_with_ffmpeg_bytes(raw: bytes) -> Tuple[torch.Tensor, int]:
    tmp_in = _bytes_to_tempfile(raw, suffix=".bin")
    tmp_out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
        _run_ffmpeg_to_wav(tmp_in, tmp_out_path, sr=SR_TARGET)
        with open(tmp_out_path, "rb") as f:
            raw_wav = f.read()
        return _read_wav_bytes(raw_wav)
    finally:
        for p in (tmp_in, tmp_out_path):
            try:
                if p:
                    os.remove(p)
            except Exception:
                pass

def _download(url: str) -> bytes:
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return r.content

# ============================================================
# MinIO helpers
# ============================================================
def _minio_client() -> Minio:
    if not (MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY):
        raise RuntimeError("MinIO not configured: set MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY.")
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)

def _parse_minio_path(path: str) -> Tuple[str, str]:
    """
    Accept:
      - minio://bucket/key
      - s3://bucket/key  (treated same)
      - bucket/key       (shorthand)
      - key (shorthand if MINIO_INPUT_BUCKET is set)
    """
    if path.startswith(("minio://", "s3://")):
        u = urlparse(path)
        bucket = u.netloc
        key = u.path.lstrip("/")
    else:
        parts = path.split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
        else:
            if not MINIO_INPUT_BUCKET:
                raise ValueError("Path must be 'minio://bucket/key', 's3://bucket/key', or 'bucket/key'.")
            bucket = MINIO_INPUT_BUCKET
            key = (MINIO_INPUT_PREFIX or "") + path
    if not bucket or not key:
        raise ValueError(f"Invalid MinIO path: {path}")
    return bucket, key

def minio_read_bytes(path: str) -> bytes:
    client = _minio_client()
    b, k = _parse_minio_path(path)
    obj = client.get_object(b, k)
    try:
        return obj.read()
    finally:
        obj.close()
        obj.release_conn()

def minio_write_bytes(path: str, data: bytes, content_type: str = "application/octet-stream"):
    client = _minio_client()
    b, k = _parse_minio_path(path)
    client.put_object(b, k, io.BytesIO(data), length=len(data), content_type=content_type)

# ============================================================
# YouTube loader with cookies/proxy/geo/etc.
# ============================================================
def _load_from_youtube(url: str, cap_seconds: Optional[float], yt_opts: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, int]:
    """
    yt_opts keys (all optional):
      - cookies_b64: base64 of Netscape cookies.txt
      - cookies_text: raw Netscape cookies file content (string)
      - cookie_path: MinIO path "minio://bucket/key" or "bucket/key"
      - proxy: e.g. "http://user:pass@host:8080"
      - geo_country: e.g. "US"
      - user_agent: UA string
      - accept_language: e.g. "en-US,en;q=0.8"
      - player_clients: list like ["android","web"]
      - no_check_cert: bool
      - format: yt-dlp format string (defaults to env YTDLP_FORMAT)
    """
    import yt_dlp

    yt_opts = yt_opts or {}
    cookie_tmp = None
    tmpdir = tempfile.mkdtemp()
    try:
        # Cookies precedence: b64 > inline text > minio path
        cookies_b64: Optional[str] = yt_opts.get("cookies_b64")
        cookies_text: Optional[str] = yt_opts.get("cookies_text")
        cookie_path: Optional[str] = yt_opts.get("cookie_path")

        if cookies_b64:
            raw = base64.b64decode(cookies_b64)
            cookie_tmp = _bytes_to_tempfile(raw, ".cookies.txt")
        elif cookies_text:
            cookie_tmp = _bytes_to_tempfile(cookies_text.encode("utf-8"), ".cookies.txt")
        elif cookie_path:
            raw = minio_read_bytes(cookie_path)
            cookie_tmp = _bytes_to_tempfile(raw, ".cookies.txt")

        headers = {
            "User-Agent": yt_opts.get("user_agent") or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": yt_opts.get("accept_language") or "en-US,en;q=0.8",
        }

        extractor_args = {"youtube": {"player_client": yt_opts.get("player_clients") or ["android", "web"]}}

        outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
        ydl_opts = {
            "format": yt_opts.get("format") or YTDLP_FORMAT,
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": YTDLP_RETRIES,
            "fragment_retries": 10,
            "skip_unavailable_fragments": True,
            "concurrent_fragment_downloads": 3,
            "http_headers": headers,
            "geo_bypass": True,
            "geo_bypass_country": yt_opts.get("geo_country"),
            "nocheckcertificate": bool(yt_opts.get("no_check_cert", False)),
            "extractor_args": extractor_args,
            "proxy": yt_opts.get("proxy"),
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
            ],
            "postprocessor_args": ["-ar", str(SR_TARGET), "-ac", str(MONO_CH)],
        }
        if cookie_tmp:
            ydl_opts["cookiefile"] = cookie_tmp
        if cap_seconds and cap_seconds > 0:
            ydl_opts["download_sections"] = {"*": [{"start_time": 0, "end_time": float(cap_seconds)}]}

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            candidate_wav = os.path.join(tmpdir, f"{info['id']}.wav")
            wav_path = os.path.join(tmpdir, "audio.wav")
            _run_ffmpeg_to_wav(candidate_wav, wav_path, sr=SR_TARGET)
            with open(wav_path, "rb") as f:
                raw_wav = f.read()
        return _read_wav_bytes(raw_wav)
    except Exception as e:
        msg = str(e)
        hint = ""
        if "403" in msg or "Forbidden" in msg or "Sign in to confirm" in msg:
            hint = (" YouTube gated the download. Provide cookies via `yt_cookies_b64`, `yt_cookies_text`, or `yt_cookie_path`; "
                    "optionally set `yt_player_clients` and `yt_geo_country`.")
        raise RuntimeError(f"YouTube download failed: {msg}.{hint}") from e
    finally:
        try:
            if cookie_tmp:
                os.remove(cookie_tmp)
        except Exception:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)

# ============================================================
# Audio I/O (local user paths disabled)
# ============================================================
def _read_audio_from_bytes(raw: bytes) -> Tuple[torch.Tensor, int]:
    try:
        data, sr = sf.read(io.BytesIO(raw), always_2d=False)
        if data.ndim == 2:
            data = data.T
        wav = torch.tensor(data, dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav, sr
    except Exception:
        return _decode_with_ffmpeg_bytes(raw)

def load_audio(
    input_audio: Any,
    youtube_cap_seconds: Optional[float] = None,
    youtube_opts: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, int]:
    """
    Accepts:
      - dict: {"type": "url"|"base64"|"path"|"data_url", "value": "..."}
      - str: url | data_url | base64 | YouTube
    'path' values are treated as MinIO paths ONLY. Local filesystem is blocked for user data.
    """
    def is_data_url(s: str) -> bool:
        return s.startswith("data:audio") and ";base64," in s

    if isinstance(input_audio, dict):
        a_type = input_audio.get("type", "url")
        value = input_audio.get("value", "")
    else:
        value = str(input_audio)
        if is_data_url(value):
            a_type = "data_url"
        elif _is_youtube_url(value):
            a_type = "url"
        elif value.startswith(("minio://", "s3://")) or ("/" in value and not value.startswith(("http://", "https://"))):
            a_type = "path"
        else:
            a_type = "url"

    if a_type == "url" and _is_youtube_url(value):
        return _load_from_youtube(value, youtube_cap_seconds, youtube_opts)

    if a_type == "url":
        raw = _download(value)
        if value.lower().endswith(".mp4"):
            return _decode_with_ffmpeg_bytes(raw)
        return _read_audio_from_bytes(raw)

    if a_type == "data_url":
        raw = base64.b64decode(value.split(",", 1)[1])
        return _read_audio_from_bytes(raw)

    if a_type == "base64":
        raw = base64.b64decode(value)
        return _read_audio_from_bytes(raw)

    if a_type == "path":
        if value.startswith(("/", "./")):
            raise RuntimeError("Local file access is disabled. Use MinIO paths like 'minio://bucket/key' or 'bucket/key'.")
        raw = minio_read_bytes(value)
        if value.lower().endswith(".mp4"):
            return _decode_with_ffmpeg_bytes(raw)
        return _read_audio_from_bytes(raw)

    raise ValueError(f"Unsupported audio type: {a_type}")

def resample_and_mono(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != SR_TARGET:
        wav = torchaudio.functional.resample(wav, sr, SR_TARGET)
    return wav

# ============================================================
# VAD utilities
# ============================================================
def _float_to_pcm16_bytes(wav: torch.Tensor) -> bytes:
    x = wav.squeeze(0).clamp_(-1.0, 1.0)
    return (x * 32767.0).to(torch.int16).cpu().numpy().tobytes()

def _frames_from_pcm16(pcm_bytes: bytes, sample_rate: int, frame_ms: int):
    n_bytes_per_sample = 2
    frame_len = int(sample_rate * frame_ms / 1000) * n_bytes_per_sample
    offset, ts, step_sec = 0, 0.0, frame_ms / 1000.0
    while offset + frame_len <= len(pcm_bytes):
        yield (ts, pcm_bytes[offset:offset + frame_len])
        offset += frame_len
        ts += step_sec

def _split_long_segments(seg: Tuple[float, float], max_len_s: float) -> List[Tuple[float, float]]:
    st, en = seg
    if en - st <= max_len_s:
        return [seg]
    spans, cur = [], st
    while cur < en:
        nxt = min(en, cur + max_len_s)
        spans.append((cur, nxt))
        cur = nxt
    return spans

def make_vad_spans(
    wav: torch.Tensor, sr: int, aggressiveness: int, frame_ms: int,
    min_speech_ms: int, max_silence_ms: int, pad_ms: int, max_segment_s: float
) -> List[Tuple[int, int]]:
    assert sr == SR_TARGET, "VAD assumes 16kHz"
    vad = webrtcvad.Vad(aggressiveness)
    pcm = _float_to_pcm16_bytes(wav.to(torch.float32))
    frames = list(_frames_from_pcm16(pcm, sr, frame_ms))
    if not frames:
        return []
    min_speech_frames = max(1, int(round(min_speech_ms / frame_ms)))
    max_silence_frames = max(1, int(round(max_silence_ms / frame_ms)))
    pad_frames = int(round(pad_ms / frame_ms))

    segments: List[Tuple[float, float]] = []
    triggered, voiced_run, silence_run = False, 0, 0
    seg_start_ts = 0.0

    for ts, fb in frames:
        is_speech = vad.is_speech(fb, sr)
        if not triggered:
            voiced_run = voiced_run + 1 if is_speech else 0
            if voiced_run >= min_speech_frames:
                triggered = True
                seg_start_ts = ts - (voiced_run - 1) * (frame_ms / 1000.0)
                voiced_run = 0
        else:
            if is_speech:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= max_silence_frames:
                    seg_end_ts = ts
                    seg_start = max(0.0, seg_start_ts - pad_frames * frame_ms / 1000.0)
                    seg_end = min(len(pcm) / (2 * sr), seg_end_ts + pad_frames * frame_ms / 1000.0)
                    for st, en in _split_long_segments((seg_start, seg_end), max_segment_s):
                        segments.append((st, en))
                    triggered, silence_run = False, 0

    if triggered:
        seg_end_ts = len(pcm) / (2 * sr)
        seg_start = max(0.0, seg_start_ts - pad_frames * frame_ms / 1000.0)
        seg_end = min(len(pcm) / (2 * sr), seg_end_ts + pad_frames * frame_ms / 1000.0)
        for st, en in _split_long_segments((seg_start, seg_end), max_segment_s):
            segments.append((st, en))

    spans: List[Tuple[int, int]] = []
    for st, en in segments:
        s, e = int(st * sr), int(en * sr)
        if e > s:
            spans.append((s, e))
    if not spans:
        spans = [(0, wav.shape[-1])]
    return spans

# ============================================================
# SRT helpers
# ============================================================
def _sec_to_srt_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _wrap_lines(text: str, max_chars: int, max_lines: int) -> str:
    words = text.strip().split()
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines[:max_lines])

def _make_chunks(n_samples: int, sr: int, chunk_s: float, overlap_s: float) -> List[Tuple[int, int]]:
    if chunk_s <= 0:
        return [(0, n_samples)]
    chunk = int(chunk_s * sr)
    step = max(1, int((chunk_s - max(0.0, overlap_s)) * sr))
    spans = []
    for st in range(0, n_samples, step):
        en = min(st + chunk, n_samples)
        if en - st > 0:
            spans.append((st, en))
        if en >= n_samples:
            break
    return spans

# Split long text into multiple SRT cues so SRT matches full text
def _split_text_even(text: str, max_chars_per_cue: int) -> List[str]:
    if len(text) <= max_chars_per_cue:
        return [text]
    words = text.split()
    parts, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars_per_cue:
            cur += " " + w
        else:
            parts.append(cur)
            cur = w
    if cur:
        parts.append(cur)
    return parts

# De-duplicate overlap between adjacent chunk transcripts
def _trim_prefix_overlap(prev_tail: str, new_text: str, max_overlap: int = 80, min_overlap: int = 12) -> str:
    prev_tail = (prev_tail or "")[-max_overlap:]
    new_text = new_text or ""
    upper = min(len(prev_tail), len(new_text), max_overlap)
    for k in range(upper, min_overlap - 1, -1):
        a = prev_tail[-k:].strip().lower()
        b = new_text[:k].strip().lower()
        if a == b and a:
            return new_text[k:].lstrip()
    return new_text

# ============================================================
# Prompt builder
# ============================================================
def build_prompt(task: str, prompt: Optional[str], source_lang: Optional[str], target_lang: Optional[str]) -> str:
    sys_prompt = (
        "Knowledge Cutoff Date: April 2024.\n"
        f"Today's Date: {time.strftime('%B %d, %Y')}.\n"
        "You are Granite, developed by IBM. You are a helpful AI assistant."
    )
    if not prompt:
        if task == "translate":
            src = (source_lang or "auto").upper()
            tgt = (target_lang or "EN").upper()
            user = f"<|audio|>Please translate the speech from {src} to {tgt}."
        else:
            user = "<|audio|>Please transcribe the speech verbatim."
    else:
        user = prompt if prompt.strip().startswith("<|audio|>") else f"<|audio|>{prompt.strip()}"
    chat = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# ============================================================
# Processor input builder (Granite-safe)
# ============================================================
def build_processor_inputs(prompt_text: str, wav_clip: torch.Tensor):
    """
    Granite Speech expects 16k mono audio as a torch.Tensor on CPU.
    Call the processor directly with (text, audio); do NOT pass sampling_rate.
    """
    audio_1d = wav_clip.squeeze(0).to(torch.float32).cpu()
    return processor(prompt_text, audio_1d, return_tensors="pt")

# ============================================================
# Core inference (single clip)
# ============================================================
@torch.inference_mode()
def transcribe_clip(prompt_text: str, wav_clip: torch.Tensor) -> Tuple[str, int]:
    inputs = build_processor_inputs(prompt_text, wav_clip)
    model_inputs = {k: (v.to(MODEL_DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
    outputs = model.generate(**model_inputs)
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = outputs[:, num_input_tokens:]
    text = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)[0]
    return text.strip(), int(new_tokens.shape[-1])

# ============================================================
# High-level entry
# ============================================================
@torch.inference_mode()
def run_inference(event_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input fields (selected):
      - audio: url | data_url | base64 | path (MinIO only) | YouTube URL
      - task: "transcribe" | "translate"
      - prompt, source_lang, target_lang
      - use_vad: bool
      - make_srt: bool
      - srt_path: MinIO path only (minio://bucket/key | bucket/key)
      - return_srt_base64: bool
      - max_new_tokens, temperature, num_beams
      - max_input_minutes: float; max_input_action: "reject"|"truncate"
      - max_youtube_minutes: float (YT pre-download cap)
      - yt_* options (cookies, proxy, geo, user-agent, player_clients, format)
    """
    t0 = time.time()

    task = (event_input.get("task") or "transcribe").lower()
    prompt_override = event_input.get("prompt")
    source_lang = event_input.get("source_lang")
    target_lang = event_input.get("target_lang", "en")
    max_new_tokens = int(event_input.get("max_new_tokens", 256))
    temperature = float(event_input.get("temperature", 0.0))
    num_beams = int(event_input.get("num_beams", 1))

    make_srt = bool(event_input.get("make_srt", False))
    srt_path = event_input.get("srt_path")
    return_srt_b64 = bool(event_input.get("return_srt_base64", RETURN_SRT_BASE64_DEFAULT))
    use_vad = bool(event_input.get("use_vad", USE_VAD_DEFAULT))

    # Per-request caps
    req_max_minutes = event_input.get("max_input_minutes")
    req_action = (event_input.get("max_input_action") or MAX_INPUT_ACTION).lower()
    req_yt_minutes = event_input.get("max_youtube_minutes")

    if "audio" not in event_input:
        raise ValueError("Missing 'audio' in input.")

    # Compute YT pre-download cap BEFORE loading audio
    yt_caps_sec = []
    if req_yt_minutes not in (None, ""):
        try:
            yt_caps_sec.append(float(req_yt_minutes) * 60.0)
        except Exception:
            pass
    if MAX_YOUTUBE_MINUTES > 0:
        yt_caps_sec.append(MAX_YOUTUBE_MINUTES * 60.0)
    if MAX_INPUT_MINUTES > 0:
        yt_caps_sec.append(MAX_INPUT_MINUTES * 60.0)
    if req_max_minutes not in (None, ""):
        try:
            yt_caps_sec.append(float(req_max_minutes) * 60.0)
        except Exception:
            pass
    if MAX_DURATION_SECONDS > 0:
        yt_caps_sec.append(MAX_DURATION_SECONDS)
    predownload_cap_sec = min(yt_caps_sec) if yt_caps_sec else None

    # Auto SRT path when requested but not provided
    if make_srt and not srt_path:
        if not MINIO_OUTPUT_BUCKET:
            raise RuntimeError("SRT requested but no MINIO_OUTPUT_BUCKET configured.")
        prefix = (MINIO_OUTPUT_PREFIX or "")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        safe_name = f"granite_{int(time.time())}.srt"
        key = f"{prefix}{safe_name}"
        srt_path = f"{MINIO_OUTPUT_BUCKET}/{key}"  # bare bucket/key shorthand

    # Gather request-level YT options
    yt_req_opts = {
        "cookies_b64":     event_input.get("yt_cookies_b64"),
        "cookies_text":    event_input.get("yt_cookies_text"),
        "cookie_path":     event_input.get("yt_cookie_path"),
        "proxy":           event_input.get("yt_proxy"),
        "geo_country":     event_input.get("yt_geo_country"),
        "user_agent":      event_input.get("yt_user_agent"),
        "accept_language": event_input.get("yt_accept_language"),
        "player_clients":  event_input.get("yt_player_clients"),
        "no_check_cert":   event_input.get("yt_no_check_cert"),
        "format":          event_input.get("yt_format"),
    }

    # Load & normalize audio
    wav, sr0 = load_audio(event_input["audio"], youtube_cap_seconds=predownload_cap_sec, youtube_opts=yt_req_opts)
    wav = resample_and_mono(wav, sr0)

    # Enforce overall length limits AFTER load
    n_samples = wav.shape[-1]
    total_seconds = n_samples / SR_TARGET

    limits = []
    if MAX_DURATION_SECONDS > 0:
        limits.append(MAX_DURATION_SECONDS)
    if MAX_INPUT_MINUTES > 0:
        limits.append(MAX_INPUT_MINUTES * 60.0)
    if req_max_minutes not in (None, ""):
        try:
            limits.append(float(req_max_minutes) * 60.0)
        except Exception:
            pass
    apply_limit = min(limits) if limits else 0.0

    if apply_limit and total_seconds > apply_limit:
        if req_action == "truncate":
            wav = wav[:, : int(apply_limit * SR_TARGET)]
            n_samples = wav.shape[-1]
            total_seconds = n_samples / SR_TARGET
        else:
            raise RuntimeError(
                f"Input audio is {total_seconds/60:.2f} min, exceeds limit of {apply_limit/60:.2f} min. "
                f"Set 'max_input_action':'truncate' to auto-trim, or lower 'max_input_minutes'."
            )

    # Build prompt
    base_prompt = build_prompt(task, prompt_override, source_lang, target_lang)

    # Generation config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else None,
        "num_beams": num_beams,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    model.generation_config.update(**gen_kwargs)

    # Spans (VAD or fixed windows)
    if use_vad:
        spans = make_vad_spans(
            wav, SR_TARGET,
            VAD_AGGRESSIVENESS, VAD_FRAME_MS,
            VAD_MIN_SPEECH_MS, VAD_MAX_SILENCE_MS,
            VAD_PAD_MS, VAD_MAX_SEGMENT_SECONDS
        )
    else:
        spans = _make_chunks(n_samples, SR_TARGET, CHUNK_SECONDS, CHUNK_OVERLAP)

    # Transcribe spans + SRT (split long text so SRT matches `text`) with overlap de-dupe
    joined_text = ""
    prev_tail = ""
    srt_entries: List[str] = []
    srt_index = 1
    total_new_tokens = 0

    for (st, en) in spans:
        clip = wav[:, st:en]
        raw_txt, new_tok = transcribe_clip(base_prompt, clip)
        total_new_tokens += new_tok

        # De-dup overlap (caused by chunk overlap or VAD padding)
        txt = _trim_prefix_overlap(prev_tail, raw_txt)
        if not txt.strip():
            continue  # skip pure duplicates

        # Accumulate plain text and update tail (keep last ~120 chars)
        joined_text = (joined_text + (" " if joined_text else "") + txt).strip()
        prev_tail = joined_text[-120:]

        # Build SRT (split long text; distribute time proportionally within span)
        start_sec, end_sec = st / SR_TARGET, en / SR_TARGET
        max_chars_per_cue = SRT_MAX_CHARS_PER_LINE * SRT_MAX_LINES
        pieces = _split_text_even(txt, max_chars_per_cue)
        total_chars = sum(len(p) for p in pieces) or 1
        cursor = start_sec
        for i, p in enumerate(pieces):
            dur = (end_sec - start_sec) * (len(p) / total_chars)
            st_s = cursor
            en_s = end_sec if i == len(pieces) - 1 else min(end_sec, cursor + dur)
            block = _wrap_lines(p, SRT_MAX_CHARS_PER_LINE, SRT_MAX_LINES)
            srt_entries.append(f"{srt_index}\n{_sec_to_srt_ts(st_s)} --> {_sec_to_srt_ts(en_s)}\n{block}\n")
            srt_index += 1
            cursor = en_s

    srt_text = "\n".join(srt_entries).strip() if (spans and (make_srt or use_vad)) else None

    wrote_path, srt_b64 = None, None
    if srt_text and srt_path:
        if srt_path.startswith(("/", "./")):
            raise RuntimeError("Local file output is disabled. Provide a MinIO path like 'minio://bucket/key' or 'bucket/key'.")
        try:
            minio_write_bytes(srt_path, srt_text.encode("utf-8"), content_type="application/x-subrip")
        except Exception as e:
            b, k = _parse_minio_path(srt_path)
            trailing = " (NOTE: key ends with '/', looks like a folder)" if k.endswith("/") else ""
            raise RuntimeError(
                f"S3 PUT failed: endpoint={MINIO_ENDPOINT}, bucket={b}, key={k}{trailing}. "
                f"Ensure your Backblaze app key has Write/List Files and the correct file-name prefix (if any). "
                f"Original error: {e}"
            ) from e
        b, k = _parse_minio_path(srt_path)
        wrote_path = f"minio://{b}/{k}"
    if srt_text and (make_srt and bool(return_srt_b64)):
        srt_b64 = base64.b64encode(srt_text.encode("utf-8")).decode("utf-8")

    t1 = time.time()
    return {
        "status": "ok",
        "text": joined_text.strip(),
        "srt": srt_text,
        "srt_base64": srt_b64,
        "srt_path_written": wrote_path,
        "timings": {"load_seconds": round(LOAD_T1 - LOAD_T0, 3), "inference_seconds": round(t1 - t0, 3)},
        "meta": {
            "device": str(MODEL_DEVICE), "dtype": str(TORCH_DTYPE), "sr_hz": SR_TARGET,
            "task": task, "model_id": MODEL_ID, "audio_seconds": round(total_seconds, 3),
            "chunks": len(spans), "use_vad": use_vad,
            "download_caps_sec": {"youtube_predownload": predownload_cap_sec},
            "vad": {
                "aggr": VAD_AGGRESSIVENESS, "frame_ms": VAD_FRAME_MS,
                "min_speech_ms": VAD_MIN_SPEECH_MS, "max_silence_ms": VAD_MAX_SILENCE_MS,
                "pad_ms": VAD_PAD_MS, "max_segment_s": VAD_MAX_SEGMENT_SECONDS
            },
            "tokens": {"new_tokens_sum": total_new_tokens}
        }
    }

# ============================================================
# Runpod entry
# ============================================================
def handler(event):
    try:
        payload = event.get("input") or {}
        return run_inference(payload)
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
