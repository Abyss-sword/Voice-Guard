FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/mpl \
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/tmp/hf GRADIO_ANALYTICS_ENABLED=False
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple/ \
    torchaudio==2.5.1 -r deploy/requirements.txt && pip check
COPY vguard/ ./vguard/
COPY scripts/ ./scripts/
COPY tools/ ./tools/
COPY frontend/ ./frontend/
COPY third_party/ada_in_vc/ ./third_party/ada_in_vc/
COPY docs/fig_eps_curve.png ./docs/fig_eps_curve.png
COPY deploy/entrypoint.py ./deploy/entrypoint.py
# Weights, registries and VCTK are mounted at runtime.
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"
ENTRYPOINT ["python", "deploy/entrypoint.py"]
