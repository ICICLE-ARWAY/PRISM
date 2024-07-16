#!/usr/bin/env python3
#
# Train a simple Convolutional Neural Network to classify image data.

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import cv2 as cv
import tensorflow as tf
#import t2onnx


def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Train a simple Convolutional Neural Network to classify image data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-n', '--num_classes', type=int, default=10, choices=[2, 10, 20, 100], help='number of classes in dataset')
    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['bf16', 'fp16', 'fp32', 'fp64'], help='floating-point precision')
    parser.add_argument('-e', '--epochs', type=int, default=42, help='number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-d', '--data_dir', type=str, default=None, help='path to data directory')
    parser.add_argument('-H', '--height', type=int, default=32, help='image height')
    parser.add_argument('-w', '--width', type=int, default=32, help='image width')
    parser.add_argument('-c', '--channels', type=int, default=3, choices=['1','3','4'], help='number of color channels')
    parser.add_argument('-l', '--load_model', type=str, default=None, help='path to model input file or directory')
    parser.add_argument('-i', '--load_format', type=str, default='keras', choices=['h5','keras', 'onnx', 'tf'], help='format of model to be loaded as input')
    parser.add_argument('-s', '--save_model', type=str, default=None, help='path to model output file or directory')
    parser.add_argument('-o', '--save_format', type=str, default='keras', choices=['h5','keras', 'onnx', 'tf'], help='format of model to be saved as output')
    parser.add_argument('-v', '--verbose', type=int, default='2', choices=[0, 1, 2], help='verbosity')

    args = parser.parse_args()
    return args

def create_datasets(data_dir, classes, height, width, channels, dtype):
    """ Create training (, validation,) and test datasets

    Two methods are currently supported:

        - keras.datasets
        - keras.utils.image_dataset_from_directory

    """

    if not data_dir:

        # Use tf.keras.datasets
        if classes == 100:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
        elif classes == 20:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
        else: # classes == 10
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Verify training and test image dataset sizes
        assert x_train.shape == (50000, 32, 32, 3)
        assert y_train.shape == (50000, 1)
        assert x_test.shape == (10000, 32, 32, 3)
        assert y_test.shape == (10000, 1)

        # Normalize the 8-bit (3-channel) RGB image pixel data between 0.0 
        # and 1.0; also converts datatype from numpy.uint8 to numpy.float64;
        # Now using tf.keras.layers.Rescaling(1./255) within the model itself
        # x_train = x_train / 255.0
        # x_test = x_test / 255.0

        # Convert from NumPy arrays to TensorFlow tensors
        x_train = tf.convert_to_tensor(value=x_train, dtype=dtype, name='x_train')
        y_train = tf.convert_to_tensor(value=y_train, dtype=tf.uint8, name='y_train')
        x_test = tf.convert_to_tensor(value=x_test, dtype=dtype, name='x_test')
        y_test = tf.convert_to_tensor(value=y_test, dtype=tf.uint8, name='y_test')

        # Construct TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    else:

        # Use tf.keras.utils.image_dataset_from_directory
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory=data_dir+'/train',
            labels='inferred',
            label_mode='int',
            color_mode='rgb',
            batch_size=None,
            image_size=(height, width),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False
        )

        test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory=data_dir+'/test',
            labels='inferred',
            label_mode='int',
            color_mode='rgb',
            batch_size=None,
            image_size=(height, width),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False
        )

        # Enforce datatypes are the same used in the datasets created by tf.keras.datasets
        train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, dtype), tf.cast(y, tf.uint8)))
        test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, dtype), tf.cast(y, tf.uint8)))

    return train_dataset, test_dataset


def create_model(height, width, channels, classes):
    """ Specify and compile the CNN model """

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(height, width, channels)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(classes),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    return model


def load_model(model_file, model_format):
    """ Load a model from file in a specific format."""
    return model


def save_model(model, model_file, model_format):
    """ Save a mode to a file in a specific format."""
    return exit_val


def main():
    """ Train CNN on CIFAR """

    # Read input variables and parse command-line arguments
    args = get_command_arguments()

    # Set internal variables from input variables and command-line arguments
    classes = args.num_classes
    if args.precision == 'bf16':
        tf_float = tf.bfloat16
    elif args.precision == 'fp16':
        tf_float = tf.float16
    elif args.precision == 'fp64':
        tf_float = tf.float64
    else: # args.precision == 'fp32'
        tf_float = tf.float32
    epochs = args.epochs
    batch_size = args.batch_size
    data_dir = args.data_dir
    height = args.height
    width = args.width
    channels = args.channels
    load_model = args.load_model
    load_format = args.load_format
    save_model = args.save_model
    save_format = args.save_format
    verbose = args.verbose

    # Create training and test datasets
    train_dataset, test_dataset = create_datasets(data_dir=data_dir, classes=classes, height=height, width=width, channels=channels, dtype=tf_float)

    # Prepare the datasets for training and evaluation
    train_dataset = train_dataset.cache().shuffle(buffer_size=50000, reshuffle_each_iteration=True).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # Create or load model
    if load_model:
        print('Load previously saved model from correct format')
    else:
        model = create_model(height=height, width=width, channels=channels, classes=classes)

    # Print summary of the model's network architecture
    model.summary()

    # Train the model on the dataset
    model.fit(x=train_dataset, epochs=epochs, verbose=verbose)

    # Evaluate the model and its accuracy
    model.evaluate(x=test_dataset, verbose=verbose)

    # Save the model
    if save_model:
        print('Save model in correct format')

    return 0


if __name__ == '__main__':
    sys.exit(main())


# References:
# https://www.tensorflow.org/tutorials/images/cnn
# https://touren.github.io/2016/05/31/Image-Classification-CIFAR10.html
# https://towardsdatascience.com/deep-learning-with-cifar-10-image-classification-64ab92110d79
# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
# https://en.wikipedia.org/wiki/8-bit_color
# https://www.tensorflow.org/guide/keras/sequential_model
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile
# https://www.tensorflow.org/guide/keras/train_and_evaluate
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#evaluate
# https://www.tensorflow.org/tutorials/load_data/images
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
# https://www.tensorflow.org/tutorials/load_data/tfrecord