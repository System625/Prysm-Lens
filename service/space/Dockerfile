# Prysm Lens search service on a Hugging Face Space (Docker SDK, CPU basic: 2 vCPU / 16 GB).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 LENS_DATA=/home/user/data
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt
# Bake the Perception Encoder weights into the image so a restart doesn't download them again.
RUN python -c "import open_clip; open_clip.create_model_and_transforms('PE-Core-L-14-336', pretrained='meta')"
COPY --chown=user . .
EXPOSE 7860
CMD ["sh", "start.sh"]
