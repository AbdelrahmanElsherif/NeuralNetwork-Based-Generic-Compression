import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os
import math
import cv2


def conv_block(input_tensor, filter):
    x = keras.layers.Conv2D(filter, (3, 3), padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def identity_block(input_tensor, filter):
    x = keras.layers.Conv2D(filter, (3, 3), padding='same', use_bias=False)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([x, input_tensor])
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Build the Mode ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Encoder part
input_shape = (32, 32, 3)
x_input = keras.layers.Input(input_shape)
x = keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), use_bias=False)(x_input) # 32x32x32 output
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.1)(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 16x16x32 output

x = identity_block(x, 32)
x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 8x8x32 output

x = conv_block(x, 16)  # 8x8x16 output
x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 4x4x16 output

x = identity_block(x, 16)
x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 2x2x16 output

x = conv_block(x, 8)  # 2x2x8 output
x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 1x1x8 output
# Compressed output here


# Decoder part
x = keras.layers.UpSampling2D((2, 2))(x)  # 2x2x8 output

x = conv_block(x, 16)  # 2x2x16 output
x = keras.layers.UpSampling2D((2, 2))(x)  # 4x4x16 output

x = identity_block(x, 16)
x = keras.layers.UpSampling2D((2, 2))(x)  # 8x8x16 output

x = conv_block(x, 32)  # 8x8x32 output
x = keras.layers.UpSampling2D((2, 2))(x)  # 16x16x32 output

x = identity_block(x, 32)
x = keras.layers.UpSampling2D((2, 2))(x)  # 32x32x32 output

x = keras.layers.Conv2D(3, (3, 3), padding='same', strides=(1, 1))(x)
x = keras.layers.Activation('sigmoid')(x)

model = keras.models.Model(inputs=x_input, outputs=x, name='ResNetIdentity')
print(model.summary())
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Compile the Model ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Import the Training & Testing Data ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Train_Data = []
Test_Data = []
Train_Count = 0
Test_Count = 0

Train_Folder = "Cropped_32x32/"
Test_Folder = "Test_32x32/"

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
---------------------------------------- Train the Model ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.fit(Train_Data,
          Train_Data,
          epochs=100,
          batch_size=256,
          shuffle=True,
          validation_data=(Test_Data, Test_Data)
          )
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Testing the Model ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

model.save("ResNetIdentity.h5")
print("Saved model to disk")