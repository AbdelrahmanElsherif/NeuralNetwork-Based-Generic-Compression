import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image


def encode_image(in_file, out_file):
    # Load the trained model
    encoder = tf.keras.models.load_model('ResNetIdentitySeperateEncoder.h5')
    print(encoder.summary())
    # Encoder Model
    #encoder = tf.keras.Model(loaded_model.layers[0], loaded_model.layers[1])

    fragments_array = []

    original_image = Image.open(in_file)
    original_image = original_image.convert('RGB')
    # Original_Image = Original_Image.convert("RGB")
    width, height = original_image.size  # Fetch image size
    # Crop input image into 32x32 blocks
    width_fragments_count = math.ceil(width / 32)  # No. of horizontal blocks
    height_fragments_count = math.ceil(height / 32)  # No. of vertical blocks
    # Setting the points for cropped image
    x1 = 0
    y1 = 0
    x2 = 32
    y2 = 32
    fragments_count = 0

    for i in range(0, height_fragments_count):

        for j in range(0, width_fragments_count):
            image_fragment = original_image.crop((x1, y1, x2, y2))  # Crop the first block
            image_fragment_array = np.array(image_fragment)  # Save it into a numpy array
            fragments_count += 1
            fragments_array.append(image_fragment_array)  # Add the first block numpy array to Input Image
            # Increment the points to catch the next fragment (16x16x3 fragments)
            x1 = x1 + 32
            x2 = x2 + 32

        x1 = 0
        y1 = y1 + 32
        x2 = 32
        y2 = y2 + 32

    # Convert input image into numpy array
    fragments_array = np.array(fragments_array)
    fragments_array = fragments_array.astype('float32') / 255

    # Output of the decoder
    prediction_encoded = encoder.predict(fragments_array)

    # Input image dimensions array
    size = [width, height]

    # Save output to the decoder and input image dimension arrays into a compressed file
    np.savez_compressed(out_file, Pred=prediction_encoded, Size=size, Type=[0])


def decode_image(in_file, out_file):
    # Load the trained model
    decoder = tf.keras.models.load_model('ResNetIdentitySeperateDecoder.h5')
    print(decoder.summary())
    # Load the compressed file
    compressed_file = np.load(in_file + ".npz")
    #print("------------------")
    #print(compressed_file.shape)
    # Load the Input to Decoder from the compressed file
    prediction = decoder.predict(compressed_file['Pred'])

    # Load the Original Image Size from the compressed file
    size_decoded = compressed_file['Size']

    # Assign the Image Size parameters to Length and Width
    width_fragments_count = math.ceil(size_decoded[0]/32)  # Width
    height_fragments_count = math.ceil(size_decoded[1]/32)  # Length

    # Initialize arrays to load the Decoded Image
    vertical_concatenated_image = []
    horizontal_concatenated_image = []
    # Initialize counter to load the Decoded Image
    fragments_count = 0

    # Loop to load and concatenate the image from the Decoder
    for i in range(0, (width_fragments_count * height_fragments_count)):
        image_fragment_array = prediction[i].reshape(32, 32, 3)  # Load 32x32 block from the decoder
        horizontal_concatenated_image.append(image_fragment_array)  # Push 32x32 block into Horizontal Concatenate Array
        fragments_count = fragments_count + 1  # Increment the Counter
        if fragments_count == width_fragments_count:  # Check if Image Width is reached by the Counter
            fragments_count = 0  # Initialize Counter
            im_h = cv2.hconcat(horizontal_concatenated_image)  # Concatenate Horizontally
            vertical_concatenated_image.append(im_h)  # Push Widthx32 blocks into Vertical Concatenate Array
            horizontal_concatenated_image.clear()  # Clear the Horizontal Concatenate Array

    reconstructed_image = cv2.vconcat(vertical_concatenated_image)  # Concatenate Vertically
    reconstructed_image = reconstructed_image[:size_decoded[1], :size_decoded[0],:] # Resize the image to it's original resolution and trim black padding
    plt.imsave(out_file, reconstructed_image)  # Save the Output Image
