FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download ONNX model at build time
RUN python -c "
from huggingface_hub import hf_hub_download
import os
model_dir = '/app/models/multilingual-e5-small'
os.makedirs(model_dir, exist_ok=True)
for fname in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.onnx']:
    hf_hub_download(
        repo_id='onnx-community/multilingual-e5-small',
        filename=fname if fname != 'model.onnx' else 'model.onnx',
        subfolder='onnx' if fname == 'model.onnx' else None,
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
"

COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]