import numpy as np
import torch
from torch.utils.data import Subset
import torchvision
from torchvision import transforms

from neural_nets_training import params


def load_dataloader(dataset, batch_size=params.batch_size):
    """ Function to create data loaders from dataset """
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=2)


def get_cifar10_dataset(dirpath="./data/CIFAR10"):
    """
    Get CIFAR 10 dataset
    Returns:
        [Tuple]: 
            (trainset, testset)
    """
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    cifar_stats = (
        CIFAR10_MEAN, CIFAR10_STD_DEV
    )  # from https://opacus.ai/tutorials/building_image_classifier

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        #transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*cifar_stats, inplace=True)
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(*cifar_stats)])

    trainset = torchvision.datasets.CIFAR10(root=dirpath,
                                            train=True,
                                            download=True,
                                            transform=train_transforms)
    testset = torchvision.datasets.CIFAR10(root=dirpath,
                                           train=False,
                                           download=True,
                                           transform=valid_transforms)
    return trainset, testset


def load_cifar10_datasets_and_loaders(dirpath="./data/CIFAR10",
                                      batch_size=params.batch_size):
    """
    Returns:
        [Tuple]: 
            (trainset, testset), (trainloader, testloader)
    """
    trainset, testset = get_cifar10_dataset(dirpath)
    trainloader, testloader = load_dataloader(
        trainset,
        batch_size=batch_size), load_dataloader(testset, batch_size=batch_size)

    return (trainset, testset), (trainloader, testloader)


def get_cifar100_dataset(dirpath="./data/CIFAR100", transforms=None):
    """Get CIFAR 100 dataset

    Args:
        dirpath (str, optional): _description_. Defaults to "./data/cifar100".
        transforms (_type_, optional): _description_. Defaults to None.
    """
    if transforms is None:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, ), (0.5, ), inplace=True),
        ])
    cifar100_train = torchvision.datasets.CIFAR100("./data/cifar100",
                                                   train=True,
                                                   download=True,
                                                   transform=transforms)
    cifar100_test = torchvision.datasets.CIFAR100("./data/cifar100",
                                                  train=False,
                                                  download=True,
                                                  transform=transforms)
    return cifar100_train, cifar100_test


def load_cifar100_datasets_and_loaders(dirpath="./data/CIFAR100",
                                       batch_size=params.batch_size,
                                       transforms=None):
    """
    Returns:
        [Tuple]: 
            (trainset, testset), (trainloader, testloader)
    """
    trainset, testset = get_cifar100_dataset(dirpath, transforms=transforms)
    trainloader, testloader = load_dataloader(
        trainset,
        batch_size=batch_size), load_dataloader(testset, batch_size=batch_size)

    return (trainset, testset), (trainloader, testloader)


def get_mnist_dataset(dirpath="./data/MNIST"):
    """Get MNIST dataset

    Args:
        dirpath (str, optional): _description_. Defaults to "./data/MNIST".

    Returns:
        tuple : (trainset, testset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ), inplace=True),
    ])

    trainset = torchvision.datasets.MNIST(dirpath,
                                          train=True,
                                          download=True,
                                          transform=transform)
    testset = torchvision.datasets.MNIST(dirpath,
                                         train=False,
                                         download=True,
                                         transform=transform)
    return trainset, testset


def load_mnist_datasets_and_loaders(dirpath="./data/MNIST",
                                    batch_size=params.batch_size):
    """
    Returns:
        [Tuple]: 
            (trainset, testset), (trainloader, testloader)
    """

    trainset, testset = get_mnist_dataset(dirpath)
    trainloader, testloader = load_dataloader(
        trainset,
        batch_size=batch_size), load_dataloader(testset, batch_size=batch_size)
    return (trainset, testset), (trainloader, testloader)


def get_data_loaders(data_loader, num_teachers, batch_size=params.batch_size):
    """ Function to create data loaders for the Teacher classifier """
    # data per teacher
    data_per_teacher = len(data_loader) // num_teachers

    return [
        torch.utils.data.DataLoader(Subset(
            data_loader,
            np.arange(0, data_per_teacher) + (data_per_teacher * teacher)),
                                    batch_size=batch_size)
        for teacher in range(num_teachers)
    ]


def dataset_from_indices(mask, dataset):
    """Generate a new dataset from a mask and a dataset.

    Args:
        mask (_type_): _description_
        dataset (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        dataset
    """
    indices = mask.nonzero()[0]
    if not (isinstance(indices, np.ndarray)):
        raise Exception(f"Indices are not numpy array: {type(indices)}")
    return Subset(dataset, indices)


def get_uneven_data_loaders(dataset, lengths):
    """ Function to create an uneven collection of dataloaders"""
    N = len(dataset)
    percent = int(N / 100)
    lengths = [
        percent * 30, percent * 20, percent * 15, percent * 10, percent * 5,
        percent * 5, percent * 5, percent * 5, percent * 5
    ]

    return torch.utils.data.random_split(dataset, lengths)
