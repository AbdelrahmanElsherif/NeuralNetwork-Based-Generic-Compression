import tensorflow as tf
import argparse
import numpy as np
from EncoderDecoderImage import encode_image, decode_image
from EncoderDecoderAudio import encode_audio, decode_audio


def encode(in_file, out_file):
    if in_file.endswith('.wav'):
        encode_audio(in_file, out_file)
    else:
        encode_image(in_file, out_file)


def decode(in_file, out_file):
    compressed_file = np.load(in_file + '.npz')
    if compressed_file['Type'] == 0:
        decode_image(in_file, out_file)
    else:
        decode_audio(in_file, out_file)


def main():

    # Do command line stuff
    parser = argparse.ArgumentParser(description='An experimental audio compressor using naive autoencoding.')
    subparsers = parser.add_subparsers(help='The mode in which to run')
    encode_parser = subparsers.add_parser('encode', help='Encode a wav file')
    encode_parser.add_argument('in_file', type=str, help='A wav file to be encoded.')
    encode_parser.add_argument('out_file', type=str,
                               help='The file path prefix for the encoded output files to be stored.')
    encode_parser.set_defaults(func=encode)
    decode_parser = subparsers.add_parser('decode', help='Decode an encoded wav file')
    decode_parser.add_argument('in_file', type=str, help='The file path prefix where the encoded files are found.')
    decode_parser.add_argument('out_file', type=str, help='The file path where the decoded wav should be stored.')
    decode_parser.set_defaults(func=decode)
    args = parser.parse_args()
    if args.func == encode or args.func == decode:
        args.func(args.in_file, args.out_file)


if __name__ == "__main__":
    main()
