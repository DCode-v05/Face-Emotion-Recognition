# Face-Detection

Webcam face detection with an emotion label drawn on each face, done three ways so the approaches can be compared: a pre-trained Keras CNN, the `deepface` library, and an OpenCV LBPH recognizer you train yourself. The repo also has a notebook of OpenCV basics and a FER-style emotion image dataset.

It started as a computer-vision learning exercise while I was working through OpenCV. The emotion weights (`emotion_model.h5`) are an external pre-trained model. The code here is the detection, preprocessing and live-feed loop around it, with no training run recorded and no accuracy numbers.

## How it works

All live scripts open the webcam with `cv2.VideoCapture(0)`, convert each frame to grayscale and run a Haar cascade `detectMultiScale` to get face rectangles. They differ after that.

- **`Face.py`** loads `emotion_model.h5`, crops each face to 48x48 grayscale, scales it to 0-1, reshapes to `(1, 48, 48, 1)` and takes the `argmax` over eight labels: surprise, neutral, sad, fear, contempt, anger, disgust, happy. It uses `face.xml` for detection.
- **`Deepface.py`** passes the face crop to `DeepFace.analyze(..., actions=['emotion'], enforce_detection=False)` and draws the `dominant_emotion`. It uses the stock cascade that ships with OpenCV (`cv2.data.haarcascades`).
- **`Train.py` and `Test.py`** are the train-it-yourself path. `Train.py` walks class folders, detects a face in each image, resizes it to 200x200, flattens it and trains `cv2.face.LBPHFaceRecognizer`, then saves `features.npy`, `labels.npy` and `face_trained_model.yml`. `Test.py` loads the model and prints and draws a label and confidence for each face in the live feed.

Press `q` to quit the live windows.

`OpenCV.ipynb` is a separate Colab notebook (about 70 cells, using `/content/Cat.jpg` and `/content/Cat_Video.mp4`, which are not in the repo). It installs `opencv-contrib-python` and `caer` and covers reading images and video, rescaling, drawing, transforms, contours, colour spaces, blurring, bitwise ops, masking, histograms, thresholding and edge detection.

## Dataset

`Dataset/` holds about 65,000 face images:

- `train/` (28,709 images) and `test/` (7,178 images), each with seven emotion folders (angry, disgust, fear, happy, neutral, sad, surprise).
- Eight class folders at the top level (anger, contempt, disgust, fear, happy, neutral, sad, surprise) with about 29,000 more images, plus `labels.csv` listing each image path and a label.

## Stack

Python, OpenCV (`opencv-python` and `opencv-contrib-python` for the LBPH recognizer), TensorFlow/Keras, `deepface`, NumPy, Jupyter/Colab.

## Setup

You need Python and a webcam.

```bash
git clone https://github.com/DCode-v05/Face-Detection.git
cd Face-Detection
pip install opencv-python opencv-contrib-python tensorflow deepface numpy
python Face.py
python Deepface.py
```

## Known problems

The LBPH scripts do not run as committed, and I have not fixed them:

- `Train.py` reads images from `Data\train` and writes to a `Faces/` folder. The dataset in this repo is in `Dataset/` and `Faces/` does not exist.
- `Test.py` reads `face_trained_model.yml` from the working directory, while `Train.py` writes it to `Faces/`.
- Both call `cv.face.LBPHFaceRecognizer()`, which is normally created with `cv.face.LBPHFaceRecognizer_create()`, and `Test.py` calls `cv.imshow(frame)` without a window name.

## Files

```
Face.py                               Haar detection + Keras CNN emotion labels
Deepface.py                           Haar detection + DeepFace emotion
Train.py, Test.py                     LBPH recognizer training and live test
OpenCV.ipynb                          OpenCV fundamentals notebook (Colab)
emotion_model.h5                      pre-trained Keras emotion model
face.xml                              Haar cascade used by Face.py, Train.py, Test.py
haarcascade_frontalface_default.xml   stock cascade, not referenced by any script
best.pt                               YOLO weights, not referenced by any script
Dataset/                              emotion images and labels.csv
```

## Contact

[Portfolio](https://www.denistan.me) · [LinkedIn](https://www.linkedin.com/in/denistanb) · [GitHub](https://github.com/DCode-v05) · denistanb05@gmail.com
