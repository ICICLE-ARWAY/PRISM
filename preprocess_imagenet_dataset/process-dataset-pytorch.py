import torch.utils.data
from torchvision import datasets
from torchvision.datasets import ImageFolder


def createdataset(root: str):
    dataset: ImageFolder = datasets.ImageFolder(root)
    trainds, testds, valds = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(6059))
    return trainds, testds, valds
