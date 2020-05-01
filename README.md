# NeuralNetwork-Based-Generic-Compression
Generic compression using autoencoder

# Installation
> Python 3.X.X is required for this to work.
1. Create your python(v3) virtual environment
2. Install the necessary packages
```
pip install tensorflow==2.1.0
pip install keras==2.3.1
pip install opencv-python==4.2.0.32
pip install Pillow==7.0.0
pip install image==1.5.28
pip install noisereduce
pip install numpy
pip install matplotlib
```
3. Download the following files from this github repo:
- [main.py](https://github.com/AbdelrahmanElsherif/NeuralNetwork-Based-Generic-Compression/blob/master/main.py)
- [EncoderDecoderAudio.py](https://github.com/AbdelrahmanElsherif/NeuralNetwork-Based-Generic-Compression/blob/master/EncoderDecoderAudio.py)
- [EncoderDecoderImage.py](https://github.com/AbdelrahmanElsherif/NeuralNetwork-Based-Generic-Compression/blob/master/EncoderDecoderImage.py)
- [audio_autoencoder.model](https://github.com/AbdelrahmanElsherif/NeuralNetwork-Based-Generic-Compression/blob/master/audio_autoencoder.model)
- [image_autoencoder.h5](https://github.com/AbdelrahmanElsherif/NeuralNetwork-Based-Generic-Compression/blob/master/image_autoencoder.h5)

##### OR

- Download this .zip file --> [Click Here](https://gofile.io/?c=hpgsf9
)
<br />
4. Put the downloaded files in a single folder, open your CMD and navigate to the folder

> Note: the model detects the filetype (image/audio) automatically, you don't have to specify.
<br />
<br />
5. To Encode (compress), use the following command:
```
python main.py encode [input_file_path] [compressed_file_path]
```
**Examples:**
**python main.py encode myimage.png mycompressed**
**python main.py encode myaudio.wav mycompressed**

> Note: You are required to include the input file extension but not the compressed file.
 <br />
 <br />
6. To Decode (decompress), use the following command:
```
python main.py decode [compressed_file_path] [output_file_path]
```
**Example:** 
**python main.py decode mycompressed my_image_output.png**
**python main.py decode mycompressed my_audio_output**
> Note: You are required to include the output file extension for the image output only.
