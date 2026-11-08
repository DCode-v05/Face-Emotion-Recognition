# Real-time Face Detection & Emotion Recognition

**Webcam face detection with on-frame emotion labels, three interchangeable backends, and an OpenCV fundamentals notebook.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![DeepFace](https://img.shields.io/badge/DeepFace-4B8BBE?style=flat) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Overview

This repo opens your webcam, finds faces in the live feed, and writes an emotion label on each one. It does the same job three ways so you can compare them: a pre-trained Keras CNN, the `deepface` library, and a classic OpenCV LBPH recognizer you train yourself. Alongside the live scripts there's `OpenCV.ipynb`, a 71-cell notebook that goes through OpenCV image-processing basics.

I built this as a computer-vision learning exercise while working through OpenCV. The emotion weights (`emotion_model.h5`) are an external pre-trained model, not trained in this repo. The code here is the detection, preprocessing and inference glue plus the live-feed loop around it. The repo also includes a FER-style emotion image dataset (about 65,000 labelled face images) for anyone who wants to retrain.

## Key Features

- **Live webcam face detection** with OpenCV Haar Cascades: each frame is converted to grayscale, `detectMultiScale` runs, and a bounding box is drawn per face.
- **Three emotion-recognition backends, each in its own script:**
  - `Face.py` loads a pre-trained Keras CNN (`emotion_model.h5`), crops each detected face to 48×48 grayscale, normalizes to 0–1, and predicts one of eight emotion labels.
  - `Deepface.py` uses the same Haar detection, but hands emotion analysis to `DeepFace.analyze(...)`, so you get the library's dominant emotion without managing a model file.
  - `Train.py` / `Test.py` are a custom **LBPH** (Local Binary Patterns Histograms) face recognizer. `Train.py` builds the feature and label arrays and writes a trained `.yml` model, and `Test.py` loads it and predicts a label and confidence on the live feed.
- **On-frame visualization**: bounding boxes plus the predicted emotion or label drawn with `cv2.putText` in an OpenCV window. Press `q` to quit.
- **OpenCV fundamentals notebook** (`OpenCV.ipynb`) covering reading images and video, rescaling, drawing shapes and text, geometric transforms, contour detection, color-space conversion, channel split/merge, blurring, bitwise operations, masking, histograms, thresholding and edge detection.
- **Bundled assets**: the pre-trained emotion CNN, two Haar Cascade XML files, a YOLO weight (`best.pt`) and the emotion dataset, so you can run inference or retrain without hunting for files.

## How It Works

### Detection, preprocessing, inference (the live pipeline)

All three live scripts share the same front end. The webcam is opened with `cv2.VideoCapture(0)` and each frame is converted to grayscale. A Haar Cascade classifier (`face.xml` in `Face.py`, `Train.py` and `Test.py`, and the stock cascade from `cv2.data.haarcascades` in `Deepface.py`) runs `detectMultiScale` to return face rectangles. After that the three scripts differ.

**Keras CNN (`Face.py`).** Each face region is cropped from the grayscale frame, resized to **48×48**, reshaped to `(1, 48, 48, 1)` and scaled to 0–1. That goes into `emotion_model.predict(...)`, and `argmax` over the output picks the label from `['surprise', 'neutral', 'sad', 'fear', 'contempt', 'anger', 'disgust', 'happy']`. The label is drawn above the face.

**DeepFace (`Deepface.py`).** Instead of running a local model, the face ROI is passed to `DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)` and the returned `dominant_emotion` is drawn on the frame. You give up some control and get convenience: no model file to manage, and DeepFace does its own preprocessing.

**LBPH recognizer (`Train.py` / `Test.py`).** This is the classical, train-it-yourself path. `Train.py` goes through class folders, detects a face in each image, resizes it to 200×200, flattens it into a feature vector, and trains `cv2.face.LBPHFaceRecognizer` on the features and labels, saving `features.npy`, `labels.npy` and `face_trained_model.yml`. `Test.py` reads the `.yml` back and, for each face in the live feed, calls `predict(...)` to get a label index and a confidence score, then prints and draws the result.

### The OpenCV notebook

`OpenCV.ipynb` is a separate learning track. It is a Google Colab notebook (it uses `from google.colab.patches import cv2_imshow`, and reads `/content/Cat.jpg` and `/content/Cat_Video.mp4`, which are not in the repo). It installs `opencv-contrib-python` and `caer`, then goes through the core OpenCV API roughly one concept per cell. It is the "learn the building blocks" companion to the scripts above, not part of the emotion pipeline.

### Dataset

`Dataset/` holds a FER-style emotion image corpus. `train/` has 28,709 images and `test/` has 7,178, each with seven emotion folders (angry, disgust, fear, happy, neutral, sad, surprise). There are also eight class folders at the top level (anger, contempt, disgust, fear, happy, neutral, sad, surprise) with about 29,000 more images, and a `labels.csv` listing image paths and labels. It's there so the LBPH recognizer, or your own CNN, can be retrained.

## Results / Highlights

There are no trained-from-scratch accuracy numbers here. The emotion CNN is an external pre-trained model and no training run is recorded in this repo. What I can state is scale and contents:

- **About 65,000 labelled face images** in the dataset (28,709 train, 7,178 test, plus the eight top-level class folders).
- **8 emotion classes** in the Keras script's label list, matching the top-level class folders.
- **3 recognition backends** (Keras CNN, DeepFace, LBPH) over one Haar-Cascade detection step.
- **71-cell OpenCV fundamentals notebook** covering about 15 image-processing topics.
- Bundled weights: `emotion_model.h5` (about 6.2 MB) and `best.pt` (about 52 MB).

## Tech Stack

- **Language:** Python 3.x (plus a Jupyter/Colab notebook)
- **Computer vision:** OpenCV (`opencv-python` / `opencv-contrib-python`): Haar Cascades, LBPH recognizer, `caer`
- **Deep learning / inference:** TensorFlow / Keras (`emotion_model.h5`), `deepface`
- **Numerics:** NumPy
- **Bundled assets:** Haar Cascade XML files, a YOLO weight (`best.pt`, included but not used by any script), FER-style emotion dataset

## Getting Started

### Prerequisites
- Python 3.x
- A working webcam (the live scripts read from camera index 0)

### Installation
```bash
git clone https://github.com/DCode-v05/Face-Emotion-Recognition.git
cd Face-Emotion-Recognition
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
The LBPH scripts do not run as committed (see Usage).

## Usage

- **Quick demo:** run `python Face.py` for the local-model path, or `python Deepface.py` if you'd rather not depend on the bundled `.h5`. Both label faces in real time.
- **Train your own recognizer:** point the LBPH scripts at a folder of class-named subfolders of face images, run `Train.py` to produce the model, then `Test.py` to try it live.
- **Learn OpenCV:** open `OpenCV.ipynb` (Colab-friendly) and run the cells to see each image-processing operation with explanatory comments.

**Known problems with the LBPH scripts** (not fixed):

- `Train.py` reads images from `Data\train` and writes to a `Faces/` folder. The dataset in this repo is in `Dataset/` and `Faces/` does not exist.
- `Test.py` reads `face_trained_model.yml` from the working directory, but `Train.py` writes it to `Faces/`.
- Both call `cv.face.LBPHFaceRecognizer()`, which is normally created with `cv.face.LBPHFaceRecognizer_create()`, and `Test.py` calls `cv.imshow(frame)` without a window name.

## Project Structure

```
Face-Emotion-Recognition/
├── Face.py                              # Live demo: Haar detection + Keras CNN emotion labels
├── Deepface.py                          # Live demo: Haar detection + DeepFace.analyze emotion
├── Train.py                             # Trains an OpenCV LBPH face recognizer (broken as committed)
├── Test.py                              # Loads the LBPH model and labels faces on the live feed (broken as committed)
├── OpenCV.ipynb                         # 71-cell OpenCV fundamentals notebook (Colab)
├── emotion_model.h5                     # Pre-trained Keras CNN for 8-class emotion (~6.2 MB)
├── best.pt                              # YOLO weight, bundled but not referenced by any script (~52 MB)
├── face.xml                             # Haar Cascade used by Face.py / Train.py / Test.py
├── haarcascade_frontalface_default.xml  # Stock Haar Cascade, not referenced by any script
├── Dataset/                             # FER-style emotion images: train/, test/, labels.csv + 8 class folders
├── .gitignore
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
