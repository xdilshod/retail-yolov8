#!/bin/bash

# Ensure the necessary directories exist
mkdir -p model_repository/models/1

echo "Downloading YOLOv8 ONNX model..."
wget -O model_repository/models/1/yolov8_best.onnx \
    "https://github.com/xdilshod/yolov8-triton/releases/download/v0.0.1/retail-yolov8-onnx.onnx"

echo "Downloading YOLOv8 PyTorch model..."
wget -O weights/retail-yolov8-model.pt \
    "https://github.com/xdilshod/yolov8-triton/releases/download/v0.0.1/retail-yolov8-model.pt"

echo "Download completed!"