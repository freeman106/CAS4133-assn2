# ASSN2 base image: CUDA 12.4 (드라이버 550 admission 통과) + torch 2.10+cu126 (vLLM 0.19.1 호환)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git build-essential ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# pip 업그레이드
RUN pip install -U pip --no-cache-dir

# torch 2.10.0+cu126 (vLLM 0.19.1 pin)
RUN pip install --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# vLLM
RUN pip install vllm==0.19.1 --no-cache-dir

# OpenRLHF 의존성 (한 줄씩, dep 충돌 회피 + layer 캐시)
RUN pip install transformers==5.7.0 --no-cache-dir
RUN pip install accelerate --no-cache-dir
RUN pip install datasets --no-cache-dir
RUN pip install peft --no-cache-dir
RUN pip install bitsandbytes --no-cache-dir
RUN pip install deepspeed==0.18.9 --no-cache-dir
RUN pip install ray[default]==2.55.0 --no-cache-dir
RUN pip install einops jsonlines loralib optimum optree pylatexenc \
                pynvml sympy tensorboard torchdata torchmetrics tqdm wheel \
                grpcio aiohttp transformers_stream_generator wandb --no-cache-dir

# HF 업로드 + papermill
RUN pip install huggingface_hub hf_transfer --no-cache-dir
RUN pip install jupyter ipykernel papermill nbconvert --no-cache-dir

WORKDIR /workspace

# vLLM 환경변수 (assn2 노트북이 사용)
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    ASSN2_VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    ASSN2_VLLM_ENFORCE_EAGER=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1
