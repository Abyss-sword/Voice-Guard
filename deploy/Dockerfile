# Voice Guard 部署镜像（华为云 ModelArts 自定义镜像 / ECS 通用）
# 构建（在仓库根目录）：
#   docker build -f deploy/Dockerfile -t voiceguard:latest .
# 本地运行：
#   docker run --gpus all -p 7860:7860 voiceguard:latest
# 说明：模型权重与注册库打进镜像（checkpoints/ + outputs/*.npz 需先就绪）。

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/tmp/hf

WORKDIR /app

# 1) Python 依赖（华为云镜像源，境内构建快）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://repo.huaweicloud.com/repository/pypi/simple/ \
    numpy scipy librosa soundfile pyyaml tqdm matplotlib gradio fastapi uvicorn \
    python-multipart gdown pypdf python-docx python-pptx openpyxl \
    speechbrain "huggingface-hub<0.30" && \
    pip install --no-cache-dir -i https://repo.huaweicloud.com/repository/pypi/simple/ \
    openai-whisper

# 2) 代码与权重
COPY vguard/ ./vguard/
COPY scripts/ ./scripts/
COPY tools/ ./tools/
COPY frontend/ ./frontend/
COPY checkpoints/ ./checkpoints/
COPY outputs/registry.npz outputs/registry_vc.npz ./outputs/
COPY data/ ./data/   # 可选：VCTK 演示音频（体积大时可去掉）

EXPOSE 7860

CMD ["python", "frontend/app.py", \
     "--model-dir", "checkpoints/adain-vc-official", \
     "--registry", "outputs/registry.npz", \
     "--attack-registry", "outputs/registry_vc.npz", \
     "--vctk", "data/VCTK22", \
     "--server-name", "0.0.0.0", "--server-port", "7860"]
