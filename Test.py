import cv2 as cv
import os

emotions=[]
DIR=r"Data/train"
for i in os.listdir(DIR):
  emotions.append(i)
print(emotions)

haar_cascade=cv.CascadeClassifier("face.xml")
face_recognizer=cv.face.LBPHFaceRecognizer()
print(type(face_recognizer))
face_recognizer.read("face_trained_model.yml")

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face_det_rec=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
    for (x,y,w,h) in face_det_rec:
        face_rec=gray[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(face_rec)
        print(f"Label is {emotions[label]} with a Confidence of {confidence:0.2f}")
        cv.putText(frame,str(emotions[label]),(0,15),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=1)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv.imshow(frame)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()