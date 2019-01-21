import cv2
import os
import numpy as np
os.environ['KERAS_BACKEND']= 'plaidml.keras.backend'
from model import create_model
model = create_model()
import matplotlib.pyplot as plt

model.load_weights('weights2.h5')  
    

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,10)

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')


while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_color = frame[y:y+h, x:x+w]

    img = cv2.resize(frame,(150,150))
    img_copy = np.copy(img)
    
    img = np.resize(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    pred = model.predict(img)
    pred = pred.reshape(-1,2)
    for i,j in pred:
        cv2.circle(img_copy,(i,j),1,(0,255,0),1)     
    try:
        cv2.imshow('Live Feed',img_copy)
    except:
        print('broken')
    if cv2.waitKey(1) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break

