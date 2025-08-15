# CUDA 12.6 + cuDNN 9 + PyTorch 2.6
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/weights/huggingface \
    HUGGINGFACE_HUB_CACHE=/weights/huggingface \
    TRANSFORMERS_CACHE=/weights/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    MODEL_ID=ibm-granite/granite-speech-3.3-8b

# System deps for audio/video
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 git ca-certificates wget \
 && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade --no-cache-dir \
      "transformers>=4.52.4" torchaudio peft soundfile accelerate \
      huggingface_hub runpod requests yt-dlp webrtcvad minio

WORKDIR /app
COPY handler.py /app/handler.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
