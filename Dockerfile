# Runpod Serverless worker: SAM2 click-to-box + YOLOv5/v8 -> ONNX export.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 SAM2_BUILD_CUDA=0 PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf YOLOV5_DIR=/opt/yolov5 YOLO_CONFIG_DIR=/tmp/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install "runpod>=1.7" "pillow>=10" "numpy<2.2" "huggingface_hub>=0.25" \
      "ultralytics==8.3.*" "onnx>=1.16,<1.18" "onnxslim>=0.1.40" "onnxruntime==1.20.*" \
      gitpython pandas seaborn scipy tqdm pyyaml requests psutil thop \
 && pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4" \
 && git clone https://github.com/ultralytics/yolov5 /opt/yolov5 && git -C /opt/yolov5 checkout af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce
# Bake the SAM2 weights into the image so cold starts do not download them.
ARG SAM2_MODEL=facebook/sam2.1-hiera-small
ENV SAM2_MODEL=${SAM2_MODEL}
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${SAM2_MODEL}')"
WORKDIR /app
COPY handler.py server.py trainer.py ./
CMD ["python", "-u", "handler.py"]
