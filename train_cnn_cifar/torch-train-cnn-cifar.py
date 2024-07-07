#%%
import argparse
import os
import sys

import keras.models
import onnx
import torch
from keras import Model
from torch import nn
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset
import torchinfo
from torchmetrics import Accuracy
import lightning as pl
import numpy as np
import onnx2keras

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
    parser.add_argument('-a', '--accelerator', type=str, default='auto', choices=['auto', 'cpu', 'gpu', 'hpu', 'tpu'], help='accelerator')
    parser.add_argument('-w', '--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('-m', '--model_file', type=str, default="", help="pre-existing model file if needing to further train model")

    args = parser.parse_args()
    return args

#%%               
def create_datasets(classes, dtype):
    """ Create CIFAR training and test datasets """

    # Download training and test image datasets
    #TODO: find a way for CIFAR-20 and fine/coarse labels
    #NOTE: DOES NOT WORK WITH CIFAR100, "target out of bounds error"
    if classes == 100:
        train_dataset = CIFAR100(root='./data', transform=ToTensor(), train=True, download=True)
        test_dataset = CIFAR100(root='./data', transform=ToTensor(), train=False, download=True)
        # cifar_dataset = pl.LightningDataModule.from_datasets(train_dataset=CIFAR100(root='./data', train=True, transform=ToTensor(), download=True),
                                                            #  test_dataset=CIFAR100(root='./data', train=False, transform=ToTensor(), download=True))
    elif classes == 20:
        # CIFAR-20 is not directly available in torchvision, this is a placeholder
        pass
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

    return train_dataset, test_dataset

#%%
class CNN(pl.LightningModule):
    def __init__(self, classes, args):
        super(CNN, self).__init__()
        self.args = args

        self.train_acc = Accuracy(num_classes=classes, task='MULTICLASS')
        self.test_acc = Accuracy(num_classes=classes, task='MULTICLASS')
        self.val_acc = Accuracy(num_classes=classes, task='MULTICLASS')

        self.cnn_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, classes)
        )

    def forward(self, x):
        return self.cnn_block(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

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

    # increase speed by optimizing fp32 matmul | TODO: MAKE THIS AN ARG
    if torch.cuda.device_count() > 0:
        torch.set_float32_matmul_precision('high')

    # Create training and test datasets
    train_dataset, test_dataset = create_datasets(classes, dtype=tf_float)

    # Prepare the datasets for training and evaluation
    cifar_datamodule = pl.LightningDataModule.from_datasets(train_dataset=train_dataset, num_workers=args.num_workers, batch_size=batch_size, val_dataset=test_dataset, test_dataset=test_dataset)

    # Create model
    if args.model_file != "":
        model = torch.load(args.model_file)
    else:
        model = CNN(classes, args)

    # Print summary of the model's network architecture
    torchinfo.summary(model, input_size=(batch_size, 3, 32, 32))

    # # Train the model on the dataset || TODO: make the accel option and devices / nodes an arg
    trainer = pl.Trainer(max_epochs=epochs, accelerator=args.accelerator)
    trainer.fit(model, datamodule=cifar_datamodule)
    trainer.test(model, dataloaders=cifar_datamodule, verbose=True)

    modelDir = "model_exports/version_" + str(trainer.logger.version)  # Create str ref of model directory
    fake_input = torch.rand((batch_size, 3, 32, 32), dtype=tf_float)  # Fake input to emulate how actual input would be given
    try:
        os.mkdir("model_exports")  # try to make model_exports folder
    except FileExistsError:
        pass
    os.mkdir(modelDir)  # make directory in model_exports for this iteration of the model

    # export ONNX and PyTorch models w/ builtin versions
    torch.onnx.export(model.eval(), fake_input, f"{modelDir}/model.onnx", input_names=["input"], output_names=["output"])
    torch.save(model.eval(), f"{modelDir}/model.pt")

    # Create Keras model from ONNX model, using that to create .keras output, .h5 (HDF5) output, and a .pb output
    # Note that .keras outputs an .h5 file as .keras is effectively a zip file containing a .h5 file along with other items
    # The addition of the explicit .h5 output is for ease-of-use
    kmodel = onnx2keras.onnx_to_keras(onnx.load_model(f"{modelDir}/model.onnx"), ["input"], name_policy="renumerate", verbose=False)
    kmodel.export(f"{modelDir}/modelprotobuf")
    keras.models.save_model(kmodel, f"{modelDir}/model.h5", save_format="h5")
    keras.models.save_model(kmodel, f"{modelDir}/model.keras", save_format="keras")

    return 0


if __name__ == '__main__':
    sys.exit(main())

# %%
