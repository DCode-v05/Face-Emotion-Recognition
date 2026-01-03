import cv2 as cv
import numpy as np
import os

emotions=[]
DIR=r"Data\train"
for i in os.listdir(DIR):
  emotions.append(i)
print(emotions)

features=[]
labels=[]
haar_cascade=cv.CascadeClassifier("face.xml")
def create_train():
    i=1
    for e in emotions:
        path=os.path.join(DIR,e)
        label=emotions.index(e)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_read=cv.imread(img_path)
            gray=cv.cvtColor(img_read,cv.COLOR_BGR2GRAY)
            face_det_rec=haar_cascade.detectMultiScale(gray,scaleFactor=1.5)
            
            for (x, y, w, h) in face_det_rec:
                print(i,"\t\t\t",x,y,w,h)
                face_det=gray[y:y+h,x:x+w]
                face_det=cv.resize(face_det,(200, 200))
                features.append(face_det.flatten())
                labels.append(label)
            i+=1
create_train()
print(f"No. of Features : {len(features)} and Labels : {len(labels)}")
features=np.array(features,dtype="object")
labels=np.array(labels)
print(type(features))
face_recognizer=cv.face.LBPHFaceRecognizer()
face_recognizer.train(features,labels)
np.save("Faces/features.npy",features)
np.save("Faces/labels.npy",labels)
face_recognizer.write("Faces/face_trained_model.yml")