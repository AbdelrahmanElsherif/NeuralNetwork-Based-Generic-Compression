import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os
import math
import cv2


def conv_block(input_tensor, kernel_size, filter, strides):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """

    x = keras.layers.Conv2D(filter, kernel_size, padding='same', strides=(1, 1), use_bias=False)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Conv2D(filter, kernel_size, padding='same', strides=strides, use_bias=False)(x)

    shortcut = keras.layers.Conv2D(filter, kernel_size, padding='same', strides=strides, use_bias=False)(input_tensor)
    x = keras.layers.add([x, shortcut])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def conv_transpose_block(input_tensor, kernel_size, filter, strides):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """

    x = keras.layers.Conv2D(filter, kernel_size, padding='same', strides=(1, 1), use_bias=False)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Conv2DTranspose(filter, kernel_size, padding='same', strides=strides, use_bias=False)(x)

    shortcut = keras.layers.Conv2DTranspose(filter, kernel_size, padding='same', strides=strides, use_bias=False)(input_tensor)
    x = keras.layers.add([x, shortcut])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
---------------------------------------- Build the Mode ----------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Encoder part
input_shape = (32, 32, 3)
x_input = keras.layers.Input(input_shape)
x = keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), use_bias=False)(x_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.1)(x)

x = conv_block(x, (3, 3), 16, (2, 2))
x = conv_block(x, (3, 3), 8, (2, 2))
x = conv_block(x, (3, 3), 8, (2, 2))
#x = conv_block(x, (3, 3), 8, (2, 2))

x = keras.layers.Conv2D(4, (3, 3), padding='same', strides=(2, 2), use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.1)(x)

# Compressed output here


# Decoder part
x = keras.layers.Conv2DTranspose(4, (3, 3), padding='same', strides=(2, 2), use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.1)(x)

#x = conv_transpose_block(x, (3, 3), 8, (2, 2))
x = conv_transpose_block(x, (3, 3), 8, (2, 2))
x = conv_transpose_block(x, (3, 3), 8, (2, 2))
x = conv_transpose_block(x, (3, 3), 16, (2, 2))

x = keras.layers.Conv2D(3, (3, 3), padding='same', strides=(1, 1))(x)
x = keras.layers.Activation('sigmoid')(x)


model = keras.models.Model(inputs=x_input, outputs=x, name='ResNetTest')
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

model.save("ResNet.h5")
print("Saved model to disk")