import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image


def encode_image(in_file, out_file):
    # Load the trained model
    loaded_model = tf.keras.models.load_model('image_autoencoder.h5')
    # Encoder Model
    encoder = tf.keras.Model(loaded_model.input, loaded_model.layers[9].output)

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
            Image_Fragment = original_image.crop((x1, y1, x2, y2))  # Crop the first block
            Image_Fragment_Array = np.array(Image_Fragment)  # Save it into a numpy array
            fragments_count += 1
            fragments_array.append(Image_Fragment_Array)  # Add the first block numpy array to Input Image
            # Increment the points to catch the next fragment (32x32x3 fragments)
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
    loaded_model = tf.keras.models.load_model('image_autoencoder.h5')

    # Decoder Model
    decoder_input = tf.keras.Input(shape=(8, 8, 2))
    decoder_layer_1 = loaded_model.layers[10]
    decoder_layer_2 = loaded_model.layers[11]
    decoder_layer_3 = loaded_model.layers[12]
    decoder_layer_4 = loaded_model.layers[13]
    decoder_layer_5 = loaded_model.layers[14]
    decoder_layer_6 = loaded_model.layers[15]
    decoder_layer_7 = loaded_model.layers[16]
    decoder_layer_8 = loaded_model.layers[17]

    decoder = tf.keras.Model(decoder_input, decoder_layer_8(
        decoder_layer_7(
            decoder_layer_6(
                decoder_layer_5(
                    decoder_layer_4(
                        decoder_layer_3(
                            decoder_layer_2(
                                decoder_layer_1(
                                    decoder_input)))))))))

    # Load the compressed file
    compressed_file = np.load(in_file + ".npz")

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
    reconstructed_image = reconstructed_image[:Size_Decoded[1], :Size_Decoded[2],:] # Resize the image to it's original resolution and trim black padding
    plt.imsave(out_file, reconstructed_image)  # Save the Output Image
