#%%
import argparse
import os
import sys
import time

import torch
import numpy as np
import keras

# %%
def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Train a simple Convolutional Neural Network to classify CIFAR images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-c', '--classes', type=int, default=10, choices=[10, 20, 100], help='number of classes in dataset')
    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['bf16', 'fp16', 'fp32', 'fp64'], help='floating-point precision')
    parser.add_argument('-e', '--epochs', type=int, default=42, help='number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')

    args = parser.parse_args()
    return args

#%%
def create_datasets(classes, dtype):
    return None, None

#%%
def create_model(classes):
    return None

#%%
def main():
    """ Train CNN on CIFAR """

    # Read input variables and parse command-line arguments
    args = get_command_arguments()




    return 0


if __name__ == '__main__':
    sys.exit(main())
