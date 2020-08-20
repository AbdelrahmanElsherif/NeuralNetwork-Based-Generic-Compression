import wave, struct
import keras
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D, AveragePooling1D, MaxPool1D, UpSampling1D, Flatten, Reshape
import numpy as np
import os
from scipy.io import wavfile

in_layer = Input(shape=(416, 1))
# Construct the encoder layers
encode = Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(in_layer)
encode = AveragePooling1D()(encode)
encode = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(encode)
encode = AveragePooling1D()(encode)
encode = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(encode)
encode = AveragePooling1D()(encode)
encode = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(encode)
encode = AveragePooling1D()(encode)
encode = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(encode)
encode = AveragePooling1D()(encode)
encode = Flatten()(encode)
encode = Dense(13, activation='relu')(encode)

# Construct the decoder layers
decode = Dense(13*64, activation='relu')(encode)
decode = Reshape((13, 64))(decode)
decode = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(decode)
decode = UpSampling1D()(decode)
decode = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(decode)
decode = UpSampling1D()(decode)
decode = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(decode)
decode = UpSampling1D()(decode)
decode = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(decode)
decode = UpSampling1D()(decode)
decode = Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(decode)
decode = UpSampling1D()(decode)
decode = Conv1D(filters=1, kernel_size=5, padding='same', activation='relu')(decode)

# The autoencoder is the whole thing
autoencoder = Model(in_layer, decode)
autoencoder.summary()

# Compile the model
autoencoder.compile('Adamax', loss='mean_squared_logarithmic_error', metrics=['accuracy'])

def load_data(DATA_FILES_WAV):
    train_data = np.array([])
    directory = os.fsencode(DATA_FILES_WAV)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".wav"):
            sample_rate, samples = wavfile.read(DATA_FILES_WAV + '/' + filename)
            samples = np.concatenate(samples)
            samples = samples.astype(float) / float(pow(2, 15))
            samples += 1.0
            samples = samples / 2.0
            samples = np.pad(samples, (0, 416-(len(samples)%416)), 'constant')
            train_data = np.append(train_data, samples)
    return train_data


cp_callback = keras.callbacks.ModelCheckpoint(filepath="cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

train_data = load_data('audio_wav_training')

autoencoder.load_weights('cp.ckpt')

train_data = train_data.reshape(len(train_data)//416 ,416, 1)
autoencoder.fit(train_data, train_data, epochs=10, shuffle=True, callbacks=[cp_callback])
autoencoder.save("autoencoder_1Channel.model")

