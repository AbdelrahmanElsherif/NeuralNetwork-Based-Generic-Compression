import wave, struct
import keras
from keras.models import Model
from keras.layers import Dense, Input, LSTM
import numpy as np
import os


DATA_FILES_WAV = 'audio_wav2'
rate = 441

in_layer = Input(shape=(1, rate))
# Construct the encoder layers
encode = LSTM(rate, activation='relu', return_sequences=True)(in_layer)
encode = Dense(rate // 2, activation='relu')(encode)
encode = Dense(rate // 4, activation='relu')(encode)
encode = Dense(rate // 8, activation='relu')(encode)

# Construct the decoder layers
decode = Dense(rate // 4, activation='relu')(encode)
decode = Dense(rate // 2, activation='relu')(decode)
decode = Dense(rate, activation='sigmoid')(decode)

# The autoencoder is the whole thing
autoencoder = Model(in_layer, decode)

# The encoder is just the first part
encoder = Model(in_layer, encode)
# And the decoder takes encoded inputs and is constructed from the latter part
encoded_input = Input(shape=(1, rate // 8))
decoder = autoencoder.layers[-3](encoded_input)
decoder = autoencoder.layers[-2](decoder)
decoder = autoencoder.layers[-1](decoder)
decoder = Model(encoded_input, decoder)
# Compile the model
autoencoder.compile('adam', loss='mse')


def dataFromWave(fname):
    """
    Reads a wav file to samples
    """
    f = wave.open(fname, 'rb')
    # Read Channel Number
    chans = f.getnchannels()
    # Get raw sample count
    samps = f.getnframes()
    # Get bit-width of samples
    sampwidth = f.getsampwidth()
    # Get sampling rate
    rate = f.getframerate()
    # Read samples
    if sampwidth == 3:  # have to read this one sample at a time
        s = ''
        for k in range(samps):
            fr = f.readframes(1)
            for c in range(0, 3 * chans, 3):
                s += '\0' + fr[c:(c + 3)]  # put TRAILING 0 to make 32-bit (file is little-endian)
    else:
        s = f.readframes(samps)
    f.close()
    # Unpack samples
    unpstr = '<{0}{1}'.format(samps * chans, {1: 'b', 2: 'h', 3: 'i', 4: 'i', 8: 'q'}[sampwidth])
    x = list(struct.unpack(unpstr, s))
    if sampwidth == 3:
        x = [k >> 8 for k in x]  # downshift to get +/- 2^24 with sign extension

    return x, chans, samps, sampwidth, rate


def train():
    train_data = np.loadtxt('train_data.csv', delimiter=':')
    autoencoder.fit(train_data, train_data, epochs=10, shuffle=True, callbacks=[cp_callback])
    autoencoder.save("autoencoder.model")


def load_data():
    train_data = np.array([[]])
    counter = 0
    directory = os.fsencode(DATA_FILES_WAV)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".wav") or filename.endswith(".py"):
            data, chans, samps, width, samp_rate = dataFromWave(DATA_FILES_WAV+"/"+filename)
            data = np.array(data)
            data = data.astype(float) / float(pow(2, 15))
            data += 1.0
            data = data / 2.0
            n_in = len(data)
            p_size = n_in + (rate - (n_in % rate))
            padded = np.zeros((p_size,))
            padded[0:n_in] = data
            inputs = padded.reshape((len(padded) // rate, rate))
            train_data = np.append(train_data, inputs)
            counter += 1
    return train_data, counter


train_data, counter = load_data()
train_data = train_data.reshape((counter*24001, 1, 441))
np.savetxt('train_data.csv', train_data, delimiter=":")
checkpoint_path = "cp.ckpt"
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
train()
