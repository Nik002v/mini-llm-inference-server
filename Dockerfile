FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_CACHE_DIR=/models
WORKDIR /service
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1' \
    && pip install -r requirements.txt \
    && useradd --create-home app && mkdir /models && chown app:app /models
COPY --chown=app:app app ./app
USER app
EXPOSE 8000
HEALTHCHECK --start-period=300s --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "150"]
