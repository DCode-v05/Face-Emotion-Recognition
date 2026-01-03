import cv2
import tensorflow as tf
import numpy as np
# Load the model
emotion_model = tf.keras.models.load_model('emotion_model.h5')
emotion_labels = ['surprise', 'neutral', 'sad', 'fear', 'contempt', 'anger', 'disgust', 'happy']

# Initialize OpenCV
cap = cv2.VideoCapture(0)  # Use webcam (index 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV Haarcascade for face detection
    face_cascade = cv2.CascadeClassifier('face.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Preprocess the face for emotion detection
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.reshape(1, 48, 48, 1) / 255.0
        
        # Predict emotion
        emotion_probabilities = emotion_model.predict(face)
        emotion_label = emotion_labels[np.argmax(emotion_probabilities)]
        
        # Display the emotion on the frame
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show the video frame
    cv2.imshow('Emotion Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
