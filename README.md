# SKU Detection with YOLOv8: ONNX Conversion and Triton Server Deployment

This repository demonstrates deploying a YOLOv8 model trained on an SKU dataset using NVIDIA's Triton Inference Server. The YOLOv8 model, converted to ONNX format, is optimized for efficient inference on both image and video data. To train SKU dataset on YOLO follow this [Ultralytics](https://docs.ultralytics.com/datasets/detect/sku-110k/#usage). You just need to change model name to yolov8.

## Directory Structure
```
.
├── LICENSE
├── README.md
├── assets
│   ├── input_img.jpg
│   └── input_video.mp4
├── download.sh
├── main.py
├── model_repository
│   └── yolo_model
│       ├── 1
│       └── config.pbtxt
├── requirements.txt
├── scripts
│   ├── __pycache__
│   │   └── general.cpython-310.pyc
│   └── general.py
├── tt2.ipynb
└── weights
```
## Features

- Triton Server Deployment: Seamless deployment of pre-trained ONNX models on NVIDIA Triton Inference Server.
- Easy-to-use Python scripts for inference.
- Dockerized Setup: Simplified Triton Server setup using Docker and organized model repository.

## Installation

#### Clone the Repository

```bash
git clone https://github.com/yourusername/yolov8-triton.git
cd yolov8-triton
```

#### Install Required Packages

```bash
pip install -r requirements.txt
```
#### Export to ONNX

If you have custom trained model you can convert to onnx format with this command
```
yolo export model=yolov8n.pt format=onnx dynamic=True opset=16
```
Rename the model file to `model.onnx` and place it under the `/model_repository/yolo_model/1` directory (see directory structure above).

## Triton Server Set-up

#### Build the Docker Container for Triton Inference

```
DOCKER_NAME="yolov8-triton"
docker build -t $DOCKER_NAME 
```

#### Run Triton Inference Server

```
DOCKER_NAME="yolov8-triton"
docker run --gpus all \
    -it --rm \
    --net=host \
    -v ./models:/models \
    $DOCKER_NAME
```

**Command Line Arguments**
```
usage: main.py [-h] --input INPUT [--output OUTPUT] [--server_url SERVER_URL]

Run inference with Triton server

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to image or video
  --output OUTPUT       Path to save output (for video inputs)
  --server_url SERVER_URL
                        Triton server URL
```

