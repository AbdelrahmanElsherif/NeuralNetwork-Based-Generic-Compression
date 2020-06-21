import keras
import numpy as np
from scipy import signal
from scipy.io import wavfile


def encode_audio(in_file, out_file):
    """
    Takes in a file path to read (a wav file)
    and a file path to write the encoded file to
    """
    # construct the encoder
    autoencoder = keras.models.load_model("audio_autoencoder.model")
    in_layer = keras.layers.Input(shape=(416, 1))
    encode = autoencoder.layers[1](in_layer)
    encode = autoencoder.layers[2](encode)
    encode = autoencoder.layers[3](encode)
    encode = autoencoder.layers[4](encode)
    encode = autoencoder.layers[5](encode)
    encode = autoencoder.layers[6](encode)
    encode = autoencoder.layers[7](encode)
    encode = autoencoder.layers[8](encode)
    encode = autoencoder.layers[9](encode)
    encode = autoencoder.layers[10](encode)
    encode = autoencoder.layers[11](encode)
    encode = autoencoder.layers[12](encode)
    encoder = keras.models.Model(in_layer, encode)

    # Read the file
    samp_rate, data = wavfile.read(in_file)
    # check if the file is mono or stereo
    if len(data.shape) == 2:
        data = np.concatenate(data)
        chans = 2
    else:
        chans = 1

    # Rescale integer samples over range [-32768,32767] to floats over range [0.0,1.0]
    data = data.astype('float32') / float(pow(2, 15))
    data += 1.0
    data = data / 2.0

    # Pad the samples with zeroes, if needed, to make the last encoding frame full
    padded = np.pad(data, (0, 416 - (len(data) % 416)), 'constant')

    # Construct input layer
    inputs = padded.reshape(len(padded) // 416, 416, 1)

    # Encode the data
    encoded = encoder.predict(inputs)

    # Save the encoded data, as well as the important parameters
    np.savez_compressed(out_file, data=encoded, rate=samp_rate, Type=1, channels=chans)


def decode_audio(in_file, out_file):
    """
    This function takes in a file prefix to a data/model file pair,
    and decodes a wav file from them at the provided location.
    """
    # construct the decoder
    autoencoder = keras.models.load_model("audio_autoencoder.model")
    in_layer = keras.layers.Input(shape=(13,))
    decode = autoencoder.layers[-13](in_layer)
    decode = autoencoder.layers[-12](decode)
    decode = autoencoder.layers[-11](decode)
    decode = autoencoder.layers[-10](decode)
    decode = autoencoder.layers[-9](decode)
    decode = autoencoder.layers[-8](decode)
    decode = autoencoder.layers[-7](decode)
    decode = autoencoder.layers[-6](decode)
    decode = autoencoder.layers[-5](decode)
    decode = autoencoder.layers[-4](decode)
    decode = autoencoder.layers[-3](decode)
    decode = autoencoder.layers[-2](decode)
    decode = autoencoder.layers[-1](decode)
    decoder = keras.models.Model(in_layer, decode)

    # Load the data
    ins = np.load(in_file + ".npz")
    encoded = ins['data']
    samp_rate = ins['rate']
    channels = ins['channels']

    # Run the decoder
    outputs = decoder.predict(encoded)

    # reform output data to the original shape and range
    out = outputs.reshape(outputs.shape[0] * outputs.shape[1])
    out = ((out * 2.0) - 1.0) * float(pow(2, 15))
    out = np.rint(out).astype(np.int16)

    # perform stft on output data to be in frequency domain
    frequencies, times, spectrogram = signal.stft(out, samp_rate, window='hann', nperseg=1024, noverlap=512)
    # eliminate values with frequencies higher than 1680 HZ to decrease noise
    spectrogram[40:, :] = 0
    # perform inverse stft to get back data in time domain
    _, out = signal.istft(spectrogram, samp_rate, window='hann', nperseg=1024, noverlap=512)
    out = np.rint(out).astype(np.int16)

    # check if file should be stereo
    if channels == 2:
        out = out.reshape(len(out)//2, 2)

    # build the wav file
    wavfile.write(out_file+'.wav', samp_rate, out)
