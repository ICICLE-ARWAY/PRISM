#!/usr/bin/env python3

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
        description='Train a simple Convolutional Neural Network to classify images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-c', '--classes', type=int, default=10, choices=[2, 10, 20, 100], help='number of classes in dataset')
    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['bf16', 'fp16', 'fp32', 'fp64'], help='floating-point precision')
    parser.add_argument('-e', '--epochs', type=int, default=42, help='number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-v', '--verbose', type=int, default='2', choices=[0, 1, 2], help='verbosity')

    parser.add_argument('-D', '--data_dir', type=str, default=None, help='path to data directory')
    parser.add_argument('-H', '--height', type=int, default=32, help='image height')
    parser.add_argument('-W', '--width', type=int, default=32, help='image width')
    parser.add_argument('-CH', '--channels', type=int, default=3, choices=['1','3','4'], help='number of color channels')

    parser.add_argument('-a', '--accelerator', type=str, default='auto', choices=['auto', 'cpu', 'gpu', 'hpu', 'tpu'], help='accelerator')
    parser.add_argument('-nw', '--num_workers', type=int, default=0, help='number of workers')

    parser.add_argument('-m', '--model_file', type=str, default="", help="pre-existing model file if needing to further train model")

    parser.add_argument('-K', '--savekeras', type=bool, default=False, help="save model as keras model file")
    parser.add_argument('-H5', '--saveh5', type=bool, default=False, help="save model as h5 model file")
    parser.add_argument('-T', '--savetensorflow', type=bool, default=False, help="save model as tf model file")
    parser.add_argument('-O', '--saveonnx', type=bool, default=False, help="save model as ONNX model file")

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
        if channels == 1:
            color_mode = 'grayscale'
        elif channels == 3:
            color_mode = 'rgb'
        else: # channels == 4
            color_mode = 'rgba'

        train_dataset = keras.preprocessing.image_dataset_from_directory(
            directory=data_dir+'/train',
            labels='inferred',
            label_mode='int',
            color_mode=color_mode,
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

        test_dataset = keras.preprocessing.image_dataset_from_directory(
            directory=data_dir+'/test',
            labels='inferred',
            label_mode='int',
            color_mode=color_mode,
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


def create_model(height, width, channels, classes, accelerator):
    """ Specify and compile the CNN model """

    if accelerator == 'auto':
        if tf.config.list_physical_devices('GPU'):
            accelerator = 'GPU'
        elif tf.config.list_physical_devices('HPU'):
            accelerator = 'HPU'
        elif tf.config.list_physical_devices('TPU'):
            accelerator = 'TPU'
        else:
            accelerator = 'CPU'
    
    accelerator = accelerator.upper()
    tf.debugging.set_log_device_placement(True)
    accelerators = tf.config.list_logical_devices(accelerator)
    strategy = tf.distribute.MirroredStrategy(accelerators)

    with strategy.scope():
        model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(height, width, channels)),
        keras.layers.Rescaling(1./255),
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

def load_model(model_file, classes, height, width, channels, accelerator):
    if model_file != "":
        model = keras.models.load_model(model_file) 
    else:
        model = create_model(height, width, channels, classes, accelerator)
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
    verbose = args.verbose
    data_dir = args.data_dir
    height = args.height
    width = args.width
    channels = args.channels
    accelerator = args.accelerator
    num_workers = args.num_workers
    model_file = args.model_file

    # Create training and test datasets
    train_dataset, test_dataset = create_datasets(data_dir, classes, height, width, channels, dtype=tf_float)

    # Prepare the datasets for training and evaluation
    train_dataset = train_dataset.cache().shuffle(buffer_size=50000, reshuffle_each_iteration=True).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # Create model
    model = load_model(model_file, classes, height, width, channels, accelerator)

    # Print summary of the model's network architecture
    model.summary()

    # Train the model on the dataset
    model.fit(x=train_dataset, epochs=epochs, workers=num_workers, verbose=verbose)

    # Evaluate the model and its accuracy
    model.evaluate(x=test_dataset, verbose=verbose)

    # Save the model in the chosen format 
    # Support for .tf, .h5, .keras, and .onnx as of now
    modelDir = "model_exports/version_tf2"
    version = os.environ.get('SLURM_JOB_ID', 'local')
    os.makedirs(modelDir, exist_ok=True)  

    if args.savetensorflow:
        # Tensorflow Format
        tf.saved_model.save(model, os.path.join(modelDir, f'{version}_model'))
    if args.saveh5:
        # HDF5 Format
        model.save(os.path.join(modelDir, f'{version}_model.h5'))
    if args.savekeras:
        # Keras format
        model.save(os.path.join(modelDir, f'{version}_model.keras'))
    if args.saveonnx:
        # ONNX format
        input_signature = [tf.TensorSpec((None, 32, 32, 3), tf.float32, name='input')]
        model.output_names=['output']
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
        onnx.save(onnx_model, f'{modelDir}/{version}_model.onnx')

    
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