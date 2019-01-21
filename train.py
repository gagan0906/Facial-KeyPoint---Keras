import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from model import create_model

images = np.load('images.npy')
key_pts = np.load('key_pts.npy')
key_pts = key_pts.reshape(-1,136)

model = create_model()
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

model.fit(images,key_pts,batch_size=20,epochs=30)

model.save_weights('new_weights.h5')