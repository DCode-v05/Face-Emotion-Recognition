# Real-time Face Detection & Emotion Recognition

## Project Description
This project implements a real-time face detection and emotion recognition system using Python and OpenCV. It leverages deep learning techniques to identify faces in a video stream (webcam) and classify the emotions of the detected individuals (e.g., happy, sad, angry, surprise). The project provides multiple implementations, including a custom TensorFlow/Keras model and an integration with the `deepface` library.

---

## Project Details

### Problem Statement
Automated facial expression recognition is a key component in human-computer interaction, used in fields ranging from market research to security. This system tackles the challenge of accurately detecting faces in real-time and mapping them to specific emotional states.

### Key Features
- **Real-time Detection:** Uses OpenCV Haar Cascades to locate faces in video frames instantly.
- **Emotion Classification:**
  - **Core Implementation:** Uses a pre-trained CNN model (`emotion_model.h5`) to classify emotions into 7 categories: surprise, neutral, sad, fear, contempt, anger, disgust, happy.
  - **DeepFace Integration:** Alternative implementation utilizing the robust `deepface` library for analysis.
- **Custom Training Support:** Includes scripts to train your own face recognizer using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm.

### Implementation Logic
1. **Face Detection:** The webcam feed is converted to grayscale. A Haar Cascade classifier identifies face regions.
2. **Preprocessing:** Detected face regions are cropped, resized to 48x48 pixels, and normalized (pixel values scaled between 0 and 1).
3. **Inference:** The processed face image is passed to the loaded Keras model, which outputs probability scores for each emotion.
4. **Visualization:** Bounding boxes and probability labels are drawn directly onto the video feed.

---

## Tech Stack
- **Languages:** Python 3.x
- **Libraries:**
  - `opencv-python` (Computer Vision)
  - `tensorflow` / `keras` (Deep Learning)
  - `deepface` (Facial Analysis)
  - `numpy` (Numerical operations)

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/DCode-v05/Face-Detection.git
cd Face-Detection
```

### 2. Install dependencies
Ensure you have Python installed, then run:
```bash
pip install opencv-python tensorflow deepface numpy
```

### 3. Run the Application
You can run the main Recogntion script:
```bash
python Face.py
```
*This will open a window showing your webcam feed with emotion labels.*

Alternatively, try the DeepFace implementation:
```bash
python Deepface.py
```

To use the custom training scripts (requires a dataset in `Data/train`):
```bash
python Train.py  # Trains the LBPH model
python Test.py   # Tests the trained LBPH model
```

---

## Project Structure
```
Face-Detection/
│
├── Face.py             # Main script: Real-time emotion recognition using Keras model
├── Deepface.py         # Alternative script: Emotion recognition using DeepFace
├── Train.py            # Script to train a custom LBPH face recognizer
├── Test.py             # Script to test the custom LBPH recognizer
├── emotion_model.h5    # Pre-trained Keras model for emotion classification
├── face.xml            # Haar Cascade XML for face detection
├── haarcascade...xml   # Additional Haar Cascade files
├── best.pt             # (Optional) PyTorch model weights
└── README.md           # Project documentation
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com
