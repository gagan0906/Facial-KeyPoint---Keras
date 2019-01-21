import cv2
import pandas as pd
import os
import numpy as np
os.environ['KERAS_BACKEND']= 'plaidml.keras.backend'
import matplotlib.pyplot as plt
from model import create_model

model = create_model()
model.load_weights('weights2.h5')   

df = pd.read_csv('../Facial Keypoints/data/test_frames_keypoints.csv')
fileNames = df.iloc[:,0].values
key_pts = df.iloc[:,1:].values

def transform(img,pts,output_size = 150):
    
    # Resizing the image and the points
    h,w = img.shape[:2]
    if h > w:
        new_h, new_w = output_size*h / w, output_size
    else:
        new_h, new_w = output_size, output_size*w /h
    
    new_h, new_w = int(new_h), int(new_w)

    img = cv2.resize(img, (new_w, new_h))
    pts = pts.reshape(-1,2)
    pts = pts * [new_w / w, new_h / h]
    
    # Cropping the image to manage the differences
    h,w= img.shape[:2]
    
    top = 0
    left = 0
    
    if h != output_size:
        top = np.random.randint(0, h - output_size)
    if w != output_size:
        left = np.random.randint(0, w - output_size)
    
    image = img[top: top + output_size,left: left + output_size]

    pts = pts - [left, top]
    
    return image, pts

print('Importing Dataset...')
training_data = []
for i,name in enumerate(fileNames):
    img = cv2.imread(os.path.join('../Facial Keypoints/data/test',name))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pts = key_pts[i]
    image, pts = transform(img,pts)
    training_data.append([image,pts])

print('Imported Successfully...')

img = [training_data[i][0] for i in range(len(training_data))]
key_pts = [training_data[i][1] for i in range(len(training_data))]

img = np.asarray(img)
key_pts = np.reshape(key_pts,(-1,136))

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
pred = model.evaluate(img,key_pts)
print(pred)
