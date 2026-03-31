# Face Mask Detection System

## Project Overview
This project provides an automated solution for public safety monitoring by identifying face mask compliance in real-time. It utilizes a deep learning approach to detect faces within a video stream and classify them based on whether a mask is being worn. The system is designed for high-performance inference on standard hardware, making it suitable for deployment in entrances, offices, and public transport hubs.

## Methodology and Architecture
The system operates in two distinct stages:

1. **Face Localization**: Utilizes a Single Shot Detector (SSD) framework with a ResNet-10 backbone to identify face coordinates within a frame.
2. **Mask Classification**: Extracts the detected face and processes it through a MobileNetV2 architecture. MobileNetV2 was selected for its efficiency in mobile and real-time vision applications, utilizing depthwise separable convolutions to maintain high accuracy with low computational overhead.

## Technical Stack
* **Programming Language**: Python 3.8+
* **Deep Learning Framework**: TensorFlow / Keras
* **Computer Vision Library**: OpenCV
* **Data Processing**: NumPy, Scikit-learn
* **Utility Libraries**: Imutils

## Project Structure
```text
.
├── dataset/
│   ├── with_mask/          # Training images with face masks
│   └── without_mask/       # Training images without face masks
├── face_detector/
│   ├── deploy.prototxt     # Model architecture for face detection
│   └── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained face detection weights
├── train_mask_detector.py  # Script for model training and serialization
├── detect_mask_video.py    # Script for real-time inference via webcam
├── mask_detector.model     # Serialized output model (Generated after training)
└── requirements.txt        # Project dependencies
```

## Setup and Installation

1. Environment Configuration
Clone the repository and install the necessary dependencies using pip:

```
Bash
pip install -r requirements.txt
```

2. Model Training
To train the classification model on the local dataset, execute:

```text
Bash
python train_mask_detector.py
This script will preprocess the images, perform data augmentation, and save the serialized model as mask_detector.model.
```

3. Real-Time Inference
To initialize the webcam and begin real-time detection, execute:

```text
Bash
python detect_mask_video.py
To terminate the video stream, press the 'q' key.
```

## Performance Analysis
The model is optimized for 224x224 input resolution, ensuring it can maintain a high frame rate (FPS) on standard hardware. By separating face detection from classification, the system reduces false positives and ensures the classifier only processes relevant image regions. The use of MobileNetV2 allows for a lightweight model file without compromising significant accuracy.
