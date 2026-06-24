# Real-time Face Detection & Emotion Recognition

**Webcam face detection with on-frame emotion labels, three interchangeable backends, plus an OpenCV fundamentals notebook.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![DeepFace](https://img.shields.io/badge/DeepFace-4B8BBE?style=flat) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Overview

This repo opens your webcam, finds faces in the live feed, and writes an emotion label on each one. It does the same job three different ways so you can compare approaches: a pre-trained Keras CNN, the `deepface` library, and a classic OpenCV LBPH recognizer you train yourself. Alongside the live scripts there's `OpenCV.ipynb` — a 56-cell notebook that walks through OpenCV image-processing fundamentals from scratch.

I built this as a computer-vision learning exercise while working through OpenCV. The emotion weights (`emotion_model.h5`) are pre-trained external models, not trained in this repo — the code here is the detection/preprocessing/inference glue and the live-feed loop around them. The repo also ships a FER-style emotion image dataset (around 36,000 labelled face images across the train/test splits) for anyone who wants to retrain.

## Key Features

- **Live webcam face detection** with OpenCV Haar Cascades — converts each frame to grayscale, runs `detectMultiScale`, and draws a bounding box per face.
- **Three emotion-recognition backends, each in its own script:**
  - `Face.py` — loads a pre-trained Keras CNN (`emotion_model.h5`), crops each detected face to 48×48 grayscale, normalizes to 0–1, and predicts one of eight emotion labels.
  - `Deepface.py` — same Haar detection, but emotion analysis is delegated to `DeepFace.analyze(...)` so you get the library's dominant-emotion output without managing a model file.
  - `Train.py` / `Test.py` — a custom **LBPH** (Local Binary Patterns Histograms) face recognizer: `Train.py` builds the feature/label arrays and writes a trained `.yml` model; `Test.py` loads it and predicts label + confidence on the live feed.
- **On-frame visualization** — bounding boxes plus the predicted emotion/label drawn with `cv2.putText`, all rendered in an OpenCV window; press `q` to quit.
- **OpenCV fundamentals notebook** (`OpenCV.ipynb`) covering reading images/video, rescaling, drawing shapes and text, geometric transforms (translate/rotate/flip/crop), contour detection, color-space conversion (BGR/RGB/Gray/HSV/LAB), channel split/merge, four blur types, bitwise operations, masking, histograms, thresholding (simple + adaptive), and gradient edge detection (Canny/Laplacian/Sobel).
- **Bundled assets** — the pre-trained emotion CNN, two Haar Cascade XML files, a YOLO weight (`best.pt`), and the full emotion dataset, so you can run inference or retrain without hunting for files.

## How It Works

### Detection → preprocessing → inference (the live pipeline)

All three live scripts share the same front end. The webcam is opened with `cv2.VideoCapture(0)` and each frame is converted to grayscale. A Haar Cascade classifier (`face.xml` or the stock `haarcascade_frontalface_default.xml`) runs `detectMultiScale` with a tuned `scaleFactor` and `minNeighbors` to return face rectangles. From there the three scripts diverge:

**Keras CNN (`Face.py`).** Each face region is cropped from the grayscale frame, resized to **48×48**, reshaped to `(1, 48, 48, 1)`, and scaled to 0–1. That tensor goes into `emotion_model.predict(...)`; `argmax` over the output picks the label from `['surprise', 'neutral', 'sad', 'fear', 'contempt', 'anger', 'disgust', 'happy']`. The label is drawn above the face.

**DeepFace (`Deepface.py`).** Instead of running a local model, the face ROI is passed to `DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)` and the returned `dominant_emotion` is drawn on the frame. This trades control for convenience — no model file to manage, and DeepFace handles its own preprocessing.

**LBPH recognizer (`Train.py` / `Test.py`).** This is the classical, train-it-yourself path. `Train.py` iterates over class folders, detects a face in each image, resizes to 200×200, flattens it into a feature vector, and trains `cv2.face.LBPHFaceRecognizer` over the collected features/labels — saving `features.npy`, `labels.npy`, and a `face_trained_model.yml`. `Test.py` reads that `.yml` back and, for each face in the live feed, calls `predict(...)` to get a label index plus a confidence score, printing and drawing the result.

### The OpenCV notebook

`OpenCV.ipynb` is a separate, self-contained learning track — a Google Colab notebook (uses `from google.colab.patches import cv2_imshow` and a sample `Cat.jpg` / `Cat_Video.mp4`). It installs `opencv-contrib-python` and `caer`, then steps through the core OpenCV API one concept per cell with inline comments explaining each parameter. It's the "learn the building blocks" companion to the application scripts above, not part of the emotion pipeline.

### Dataset

`Dataset/` holds a FER-style emotion image corpus: a `train/` split (~28.7k images), a `test/` split (~7.2k images), a `labels.csv`, and eight emotion-named class folders (anger, contempt, disgust, fear, happy, neutral, sad, surprise) with roughly 26k more images between them. It's there so the LBPH recognizer (or your own CNN) can be retrained.

## Results / Highlights

There are no trained-from-scratch accuracy numbers here — the emotion CNN is a pre-trained external model and no training run is recorded in this repo. The figures that are real are about scale and contents:

- **~36,000 labelled face images** across the dataset (≈28.7k train / ≈7.2k test) plus eight emotion class folders.
- **8 emotion classes** declared by the Keras script's label list and matched by the dataset's class folders.
- **3 independent recognition backends** (Keras CNN, DeepFace, LBPH) over one shared Haar-Cascade detector.
- **56-cell OpenCV fundamentals notebook** covering ~15 distinct image-processing topics.
- Repo size ~470 MB, dominated by the dataset and bundled weights (`emotion_model.h5` ≈ 6.2 MB, `best.pt` ≈ 52 MB).

## Tech Stack

- **Language:** Python 3.x (plus a Jupyter/Colab notebook)
- **Computer vision:** OpenCV (`opencv-python` / `opencv-contrib-python`) — Haar Cascades, LBPH recognizer, `caer`
- **Deep learning / inference:** TensorFlow / Keras (`emotion_model.h5`), `deepface`
- **Numerics:** NumPy
- **Bundled assets:** Haar Cascade XML files, a YOLO weight (`best.pt`, included but not wired into any script), FER-style emotion dataset

## Getting Started

### Prerequisites
- Python 3.x
- A working webcam (the live scripts read from camera index 0)

### Installation
```bash
git clone https://github.com/DCode-v05/Face-Detection.git
cd Face-Detection
pip install opencv-python opencv-contrib-python tensorflow deepface numpy
```

### Running
```bash
# Pre-trained Keras CNN backend
python Face.py

# DeepFace backend
python Deepface.py
```
Each opens a window showing your webcam feed with emotion labels drawn on detected faces. Press `q` to quit.

For the LBPH recognizer (train, then test):
```bash
python Train.py   # builds and saves the LBPH model
python Test.py    # loads the model and labels faces live
```

## Usage

- **Quick demo:** run `python Face.py` for the local-model path or `python Deepface.py` if you'd rather not depend on the bundled `.h5`. Both label faces in real time.
- **Train your own recognizer:** point the LBPH scripts at a folder of class-named subfolders of face images, run `Train.py` to produce the model, then `Test.py` to evaluate it live.
- **Learn OpenCV:** open `OpenCV.ipynb` (Colab-friendly) and run the cells to see each image-processing operation with explanatory comments.

A few things to know before you run the training path (see Notes).

## Project Structure

```
Face-Detection/
├── Face.py                              # Live demo: Haar detection + Keras CNN emotion labels
├── Deepface.py                          # Live demo: Haar detection + DeepFace.analyze emotion
├── Train.py                             # Trains an OpenCV LBPH face recognizer
├── Test.py                              # Loads the LBPH model and labels faces on the live feed
├── OpenCV.ipynb                         # 56-cell OpenCV fundamentals notebook (Colab)
├── emotion_model.h5                     # Pre-trained Keras CNN for 8-class emotion (~6.2 MB)
├── best.pt                              # YOLO weight, bundled but not referenced by any script (~52 MB)
├── face.xml                             # Haar Cascade used by Face.py / Train.py / Test.py
├── haarcascade_frontalface_default.xml  # Stock Haar Cascade used by Deepface.py
├── Dataset/                             # FER-style emotion images: train/, test/, labels.csv + 8 class folders
└── README.md
```

---

## Contact

**Portfolio:** [Denistan](https://www.denistan.me)<br>
**LinkedIn:** [Denistan](https://www.linkedin.com/in/denistanb)<br>
**GitHub:** [DCode-v05](https://github.com/DCode-v05)<br>
**LeetCode:** [Denistan_B](https://leetcode.com/u/Denistan_B)<br>
**Email:** [denistanb05@gmail.com](mailto:denistanb05@gmail.com)

Made with ❤️ by **Denistan B**
