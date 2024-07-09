#!/usr/bin/env python3
# NOTE: USES KERAS3 API

import argparse
import os
import sys

import tensorflow as tf
import keras
import onnx
import tf2onnx

def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Train a simple Convolutional Neural Network to classify CIFAR images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-c', '--classes', type=int, default=10, choices=[10, 100], help='number of classes in dataset')
    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['bf16', 'fp16', 'fp32', 'fp64'], help='floating-point precision')
    parser.add_argument('-e', '--epochs', type=int, default=42, help='number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-a', '--accelerator', type=str, default='auto', choices=['auto', 'cpu', 'gpu', 'hpu', 'tpu'], help='accelerator')
    parser.add_argument('-w', '--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('-m', '--model_file', type=str, default="", help="pre-existing model file if needing to further train model")
    parser.add_argument('-s', '--save_model', type=str, default="onnx", choices=["onnx", "pt"], help="save model as ONNX or PyTorch model file")

    args = parser.parse_args()
    return args


def create_datasets(classes, dtype):
    """ Create CIFAR training and test datasets """

    # Download training and test image datasets
    if classes == 100:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')
    elif classes == 20:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='coarse')
    else: # classes == 10
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Verify training and test image dataset sizes
    assert x_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_test.shape == (10000, 1)

    # Normalize the 8-bit (3-channel) RGB image pixel data between 0.0 
    # and 1.0; also converts datatype from numpy.uint8 to numpy.float64
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert from NumPy arrays to TensorFlow tensors
    x_train = tf.convert_to_tensor(value=x_train, dtype=dtype, name='x_train')
    y_train = tf.convert_to_tensor(value=y_train, dtype=tf.uint8, name='y_train')
    x_test = tf.convert_to_tensor(value=x_test, dtype=dtype, name='x_test')
    y_test = tf.convert_to_tensor(value=y_test, dtype=tf.uint8, name='y_test')

    # Construct TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train_dataset, test_dataset


def create_model(classes, args):
    """ Specify and compile the CNN model """

    if args.accelerator == 'auto':
        if tf.config.list_physical_devices('GPU'):
            args.accelerator = 'GPU'
        elif tf.config.list_physical_devices('HPU'):
            args.accelerator = 'HPU'
        elif tf.config.list_physical_devices('TPU'):
            args.accelerator = 'TPU'
        else:
            args.accelerator = 'CPU'
    
    args.accelerator = args.accelerator.upper()
    tf.debugging.set_log_device_placement(True)
    accelorators = tf.config.list_logical_devices(args.accelerator)
    strategy = tf.distribute.MirroredStrategy(accelorators)

    with strategy.scope():
        model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(classes),
        ])
         
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    return model


def main():
    """ Train CNN on CIFAR """

    # Read input variables and parse command-line arguments
    args = get_command_arguments()

    # Set internal variables from input variables and command-line arguments
    classes = args.classes
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

    # Create training and test datasets
    train_dataset, test_dataset = create_datasets(classes, dtype=tf_float)

    # Prepare the datasets for training and evaluation
    train_dataset = train_dataset.cache().shuffle(buffer_size=50000, reshuffle_each_iteration=True).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # Create model
    if args.model_file != "":
        model = keras.models.load_model(args.model_file) 
    else:
        model = create_model(classes)

    # Print summary of the model's network architecture
    model.summary()

    # Train the model on the dataset
    model.fit(x=train_dataset, epochs=epochs, verbose=2)

    # Evaluate the model and its accuracy
    model.evaluate(x=test_dataset, verbose=2)

    # Save the model in the chosen format 
    # Support for .tf, .h5, .keras, and .onnx as of now
    # Save the model in the chosen format
    model_dir = 'saved_model_tf' + os.environ.get('SLURM_JOB_ID', 'local')
    os.makedirs(model_dir, exist_ok=True)
    

    if args.model_file == 'tf':
        # Native Tensorflow format
        tf.saved_model.save(model, os.path.join(model_dir, 'model_tf'))


    elif args.model_file == 'h5':
        # HDF5 Format
        model.save(os.path.join(model_dir, 'model_tf.h5'))

    elif args.model_file == 'keras':
        # Keras format
        model.save(os.path.join(model_dir, 'model_tf.keras'))
    else:
        # ONNX format
        model.save(os.path.join(model_dir, 'model_tf.onnx'))

    
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