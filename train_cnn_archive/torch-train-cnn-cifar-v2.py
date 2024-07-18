#%%
import argparse
import os
import sys

import torch
import pandas as pd
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset
from torchmetrics import Accuracy
import lightning as pl
import numpy as np

import cv2 as cv

# %%
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
    parser.add_argument('-v', '--verbose', type=int, default='True', choices=['True', 'False'], help='display verbosity')

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

#%%               
def create_datasets(data_dir, classes, height, width, channels, dtype):
    """ Create training (, validation,) and test datasets
    """

    if not data_dir:

        # Download training and test image datasets
        if classes == 100:
            train_dataset = CIFAR100(root='./data', transform=ToTensor(), train=True, download=True)
            test_dataset = CIFAR100(root='./data', transform=ToTensor(), train=False, download=True)
        else: # classes == 10
            train_dataset = CIFAR10(root='./data', transform=ToTensor(), train=True, download=True)
            test_dataset = CIFAR10(root='./data', transform=ToTensor(), train=False, download=True)

        # Converting to respective tensors for analysis and building datasets
        x_train, y_train = np.array(train_dataset.data), np.array(train_dataset.targets)
        x_test, y_test = np.array(test_dataset.data), np.array(test_dataset.targets)

        # Verify training and test image dataset sizes
        assert x_train.shape == (50000, 32, 32, 3)
        assert y_train.shape == (50000,)
        assert x_test.shape == (10000, 32, 32, 3)
        assert y_test.shape == (10000,)

        # Normalize the 8-bit (3-channel) RGB image pixel data between 0.0 
        # and 1.0; also converts datatype from numpy.uint8 to numpy.float64
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Convert from NumPy arrays to PyTorch tensors
        x_train = torch.tensor(data=x_train, dtype=dtype).permute(0, 3, 1, 2)
        y_train = torch.tensor(data=y_train, dtype=torch.uint8)
        x_test = torch.tensor(data=x_test, dtype=dtype).permute(0, 3, 1, 2)
        y_test = torch.tensor(data=y_test, dtype=torch.uint8)

        # Construct PyTorch datasets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
    
    else:
        train_dataset = CustomDataset(csv_file=data_dir + "/train.csv", root_dir=data_dir + "/train", height=height, width=width, channels=channels, transform=ToTensor())
        test_dataset = CustomDataset(csv_file=data_dir + "/test.csv", root_dir=data_dir + "/test", height=height, width=width, channels=channels, transform=ToTensor())

    return train_dataset, test_dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, height, width, channels, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.height = height
        self.width = width
        self.channels = channels

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


#%%
class CNN(pl.LightningModule):
    def __init__(self, classes, args):
        super(CNN, self).__init__()
        self.args = args

        self.train_acc = Accuracy(num_classes=classes, task='MULTICLASS')
        self.test_acc = Accuracy(num_classes=classes, task='MULTICLASS')
        self.val_acc = Accuracy(num_classes=classes, task='MULTICLASS')

        #tensor format: [batch_size, channels, height, width]
        self.cnn_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 64),
            #torch.nn.Linear(linear_shape, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, classes)
        )

    def forward(self, x):
        return self.cnn_block(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.log("train_acc_epoch", self.train_acc.compute(), prog_bar=True, on_epoch=True)

        self.train_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer
    
def load_model(model_file, classes, args):
    if model_file != "":
        model = torch.load(model_file)
    else:
        model = CNN(classes, args)

    return model

#%%
def main():
    """ Train CNN on CIFAR """

    # Read input variables and parse command-line arguments
    args = get_command_arguments()

    # Set internal variables from input variables and command-line arguments
    classes = args.classes
    match args.precision:
        case 'bf16': tf_float = torch.bfloat16
        case 'fp16': tf_float = torch.float16
        case 'fp64': tf_float = torch.float64
        case 'fp32': tf_float = torch.float32
        case _: raise Exception(
                "Provided precision string: " +
                args.precision +
                " is not within the accepted set of values: ['bf16', 'fp16', 'fp64', 'fp32']"
            )
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

    # increase speed by optimizing fp32 matmul | TODO: MAKE THIS AN ARG
    if torch.cuda.device_count() > 0:
        torch.set_float32_matmul_precision('high')

    # Create training and test datasets
    train_dataset, test_dataset = create_datasets(data_dir, classes, height, width, channels, dtype=tf_float)

    # Prepare the datasets for training and evaluation
    cifar_datamodule = pl.LightningDataModule.from_datasets(train_dataset=train_dataset, num_workers=num_workers, batch_size=batch_size, val_dataset=test_dataset, test_dataset=test_dataset)

    # Create model
    model = load_model(model_file, classes, args)

    # # Train the model on the dataset || TODO: make the accel option and devices / nodes an arg
    trainer = pl.Trainer(max_epochs=epochs, accelerator=accelerator)
    trainer.fit(model, datamodule=cifar_datamodule)
    trainer.test(model, dataloaders=cifar_datamodule, verbose=verbose)

    fake_input = torch.rand((batch_size, 3, 32, 32), dtype=tf_float)  # Fake input to emulate how actual input would be given

    modelDir = "model_exports/version_torch"  # Create str ref of model directory
    version = str(trainer.logger.version)

    os.makedirs(modelDir, exist_ok = True) 

    # export ONNX and PyTorch models w/ builtin versions
    if args.saveonnx:
        torch.onnx.export(model.eval(), fake_input, f"{modelDir}/{version}_model.onnx", input_names=["input"], output_names=["output"])
    if args.savepytorch:
        torch.save(model.eval(), f"{modelDir}/{version}_model.pt")

    return 0


if __name__ == '__main__':
    sys.exit(main())

# %%
