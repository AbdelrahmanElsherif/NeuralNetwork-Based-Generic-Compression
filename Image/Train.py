import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os
import math
import cv2

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Import the Training & Testing Data ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Train_Data = []
Test_Data = []
Train_Count = 0
Test_Count = 0

Train_Folder = "Cropped/"
Test_Folder = "Test/"

for filename in os.listdir(Train_Folder):
    img = cv2.imread(os.path.join(Train_Folder, filename), cv2.IMREAD_COLOR)
    if img is not None:
        Train_Count = Train_Count + 1
        Train_Data.append(img)

for filename in os.listdir(Test_Folder):
    img = cv2.imread(os.path.join(Test_Folder, filename), cv2.IMREAD_COLOR)
    if img is not None:
        Test_Count = Test_Count + 1
        Test_Data.append(img)

Train_Data = np.array(Train_Data)
Test_Data = np.array(Test_Data)

Max_Value = float(Train_Data.max())
Train_Data = Train_Data.astype('float32') / 255
Test_Data = Test_Data.astype('float32') / 255

Train_Data = Train_Data.reshape((Train_Count, 32, 32, 3))
Test_Data = Test_Data.reshape((Test_Count, 32, 32, 3))
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Build the Mode ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Autoencoder = keras.Sequential()

# Encoder Layers
Autoencoder.add(keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3)))
Autoencoder.add(keras.layers.Activation('relu'))
Autoencoder.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

# 2nd convolution layer
Autoencoder.add(keras.layers.Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
Autoencoder.add(keras.layers.Activation('relu'))
Autoencoder.add(keras.layers.Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
Autoencoder.add(keras.layers.Activation('relu'))
Autoencoder.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))


# here compressed version


# 3rd convolution layer
Autoencoder.add(keras.layers.Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
Autoencoder.add(keras.layers.Activation('relu'))
Autoencoder.add(keras.layers.Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
Autoencoder.add(keras.layers.Activation('relu'))
Autoencoder.add(keras.layers.UpSampling2D((2, 2)))

# 4rd convolution layer
Autoencoder.add(keras.layers.Conv2D(16, (3, 3), padding='same'))
Autoencoder.add(keras.layers.Activation('relu'))
Autoencoder.add(keras.layers.UpSampling2D((2, 2)))

Autoencoder.add(keras.layers.Conv2D(3, (3, 3), padding='same'))
Autoencoder.add(keras.layers.Activation('sigmoid'))


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Compile the Model ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Train the Model ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Autoencoder.fit(Train_Data,
                Train_Data,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                validation_data=(Test_Data, Test_Data)
                )

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Testing the Model ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Autoencoder.save("Autoencoder1000.h5")
print("Saved model to disk")
