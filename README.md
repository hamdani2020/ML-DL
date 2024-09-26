# GOKAC GeoPothole Application

The aim of this project is to use deep learning algoritms to detect potholes in real-time. This project is powered by YOLOv8.

## Homepage
![image](resource/homepage.png)

## Features
- Image detection and segmentation
- Video detection
- Real-time Detection
- Geotag and Map
- Volume and Area Computation
- Local Address


## Running on Local Server

This is the step that you take to install and run the web-application on the local server.

``` bash
# Install CUDA if available
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

# Create the python environment

# Install pytorch-CUDA
# https://pytorch.org/get-started/locally/
pip install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install ultralytics deep learning framework
# https://docs.ultralytics.com/quickstart/
pip install ultralytics

# Clone the repository


# Install requirements
pip install -r requirements.txt

# Start the streamlit webserver
streamlit run Home.py
```

## LIVE DEMO
[GOKAC GeoPothole](https://hocaipt.streamlit.app/)

The webserver illustration is available on the streamlit cloud. However, certain functionality might not be functioning as intended owing to hardware restrictions. For example, the webcam input cannot be captured by the real-time detection, and the video detection inference is slow.

## Evaluation Results

![image](resource/MaskPR_curve.png)
![image](resource/MaskP_curve.png)
![image](resource/confusion_matrix.png)
![image](resource/val_batch2_pred.jpg)
