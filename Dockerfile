FROM python:3.11-slim

WORKDIR /app

# Зависимости отдельным слоем (кэшируется). Torch берётся в CPU-варианте, без CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY Task4/ Task4/

ENV HF_HOME=/hf_cache \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning

ENTRYPOINT ["python", "Task4/chat.py"]
