#%%
import argparse
import os
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
import torchsummary
from torchmetrics.functional import accuracy
import lightning as pl
import numpy as np

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
class CNN(pl.LightningModule):
    def __init__(self, classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = accuracy(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

class CIFARDATA(pl.LightningDataModule):
    def __init__(self, batch_size, classes=100):
        super().__init__()
        self.batch_size = batch_size
        self.classes = classes
        self.transform = ToTensor()

    def prepare_data(self):
        # Download data if needed
        CIFAR100(root='./data', train=True, download=True)
        CIFAR100(root='./data', train=False, download=True)

    def setup(self, stage=None):
        # Transform data and prepare train/test datasets
        if self.classes == 100:
            self.train_dataset = CIFAR100(root='./data', train=True, transform=self.transform)
            self.test_dataset = CIFAR100(root='./data', train=False, transform=self.transform)
        elif self.classes == 20:
            # Placeholder for CIFAR-20
            pass
        else:  # classes == 10
            self.train_dataset = CIFAR100(root='./data', train=True, transform=self.transform)
            self.test_dataset = CIFAR100(root='./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
                                
def create_datasets(classes, dtype):
    """ Create CIFAR training and test datasets """

    # Download training and test image datasets
    #TODO: find a way for CIFAR-20 and fine/coarse labels
    #ToTensor() converts to 0.0-1.0, float32
    transform = ToTensor()

    if classes == 100:
        train_dataset = CIFAR100(root='./data', train=True, transform=transform, download=True)
        test_dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)
    elif classes == 20:
        # CIFAR-20 is not directly available in torchvision, this is a placeholder
        pass
    else: # classes == 10
        train_dataset = CIFAR100(root='./data', train=True, transform=transform, download=True)
        test_dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)

    #Converting to respective tensors for analysis and building datasets
    x_train, y_train = torch.tensor(data=train_dataset.data, dtype=dtype), torch.tensor(data=train_dataset.targets, dtype=torch.uint8)
    x_test, y_test = torch.tensor(data=test_dataset.data, dtype=dtype), torch.tensor(data=test_dataset.targets, dtype=torch.uint8)

    # Verify training and test image dataset sizes
    # assert x_train.size == (50000, 32, 32, 3)
    # assert y_train.size == (50000, 1)
    # assert x_test.size == (10000, 32, 32, 3)
    # assert y_test.size == (10000, 1)



    # Construct TensorFlow datasets
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    return train_dataset, test_dataset

#%%
# def create_model(classes):
#     """ Specify and compile the CNN model """

#     #NOTE: could not find InputLayer equivalent in torch, so assume that the size is 32x32, input channels specified in first Conv2d
#     #NOTE: torch does not have activation incorporated, have to call seperately
#     #NOTE: torch equivalent of Dense layer is Linear
#     model = nn.Sequential(
#         nn.Conv2d(in_channels=3, filters=32, kernel_size=3),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2),
#         nn.Conv2d(filters=64, kernel_size=3),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2),
#         nn.Conv2d(filters=64, kernel_size=3),
#         nn.ReLU(),
#         nn.Flatten(),
#         nn.Linear(64,64),
#         nn.ReLU(),
#         nn.Linear(64,classes)
#     )

#     model.compile(
#         optimizer=torch.optim.Adam(),
#         loss=nn.CrossEntropyLoss(),
#         metrics=['accuracy'],
#     )

#     return model

#%%
def main():
    """ Train CNN on CIFAR """

    # Read input variables and parse command-line arguments
    args = get_command_arguments()

    # Set internal variables from input variables and command-line arguments
    classes = args.classes
    if args.precision == 'bf16':
        tf_float = torch.bfloat16
    elif args.precision == 'fp16':
        tf_float = torch.float16
    elif args.precision == 'fp64':
        tf_float = torch.float64
    else: # args.precision == 'fp32'
        tf_float = torch.float32
    epochs = args.epochs
    batch_size = args.batch_size

    # Create training and test datasets
    train_dataset, test_dataset = create_datasets(classes, dtype=tf_float)

    # Prepare the datasets for training and evaluation
    #TODO: cache
    data = CIFARDATA(batch_size, classes)

    # Create model
    model = CNN(classes)

    # Print summary of the model's network architecture
    torchsummary.summary(model, input_size=(32, 32, 3))

    # Train the model on the dataset
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, datamodule=data)
    
    # trainer.test(model, test_dataloaders = test_dataset)

    trainer.save_checkpoint('saved_model_'+"""os.environ['SLURM_JOB_ID']""" "hello" + ".ckpt")

    return 0


if __name__ == '__main__':
    sys.exit(main())

# %%
