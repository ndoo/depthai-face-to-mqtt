# Face Recognition to MQTT

## Goals

- [x] Facial recognition with DNN running on [Luxonis OAK-D-PoE](https://shop.luxonis.com/products/oak-d-poe)
- [ ] Publish recognition to MQTT broker
- [ ] Publish [Home Assistant Device Tracker discovery](https://www.home-assistant.io/integrations/device_tracker.mqtt/#discovery-schema) to MQTT broker
- [ ] Subscribe to MQTT broker for enrolment

## Enhancements

- [ ] Parallel pipeline to enrol/recognize with depth

## How it Works

Based on [DepthAI's gen2-face-recognition experiment](https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-recognition).

### Model Pipeline

1. [face-detection-retail-0004](models/face-detection-retail-0004_openvino_2020_1_4shave.blob) model to identify face
2. [head-pose-estimation-adas-0001](models/head-pose-estimation-adas-0001.blob) model to straighten head posture
3. [face-recognition-mobilefacenet-arcface.blob](models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob) model to recognize face and give confidence (cosine similarity) to enrolled DNN data

## Pre-requisites

1. [Purchase a DepthAI camera](https://shop.luxonis.com/)
2. Install requirements
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## Usage

1. First use - enroll face with
   ```bash
   python3 main.py --debug --enroll [name]
   ```
2. Test recognition
   ```bash
   python3 main.py --debug
   ```

> Press 'q' to exit the program.