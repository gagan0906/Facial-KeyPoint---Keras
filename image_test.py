import cv2
import os
import numpy as np
os.environ['KERAS_BACKEND']= 'plaidml.keras.backend'
from model import create_model
model = create_model()
import matplotlib.pyplot as plt

model.load_weights('weights2.h5')  
    
def show(img, pts):
    plt.imshow(img)
    plt.scatter(pts[:,0],pts[:,1],marker='.')
    plt.show()

img = cv2.imread('image.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(150,150))

img_copy = np.copy(img)
img = np.reshape(img, (1,img.shape[0],img.shape[1],img.shape[2]))

pred = model.predict(img)

pred = pred.reshape(-1,2)

show(img_copy,pred)
 
