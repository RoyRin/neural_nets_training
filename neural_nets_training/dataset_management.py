import numpy as np
import torch
from torch.utils.data import Subset
import torchvision
from torchvision import transforms

from neural_nets_training import params


def load_dataloader(dataset, batch_size=params.batch_size):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=2)


def get_CIFAR10_dataset(dirpath="./data/CIFAR10"):
    """
    Returns:
        [Tuple]: 
            (trainset, testset), (trainloader, testloader)
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


def load_CIFAR10_datasets_and_loaders(dirpath="./data/CIFAR10",
                                      batch_size=params.batch_size):
    """
    Returns:
        [Tuple]: 
            (trainset, testset), (trainloader, testloader)
    """
    trainset, testset = get_CIFAR10_dataset(dirpath)
    trainloader, testloader = load_dataloader(
        trainset,
        batch_size=batch_size), load_dataloader(testset, batch_size=batch_size)

    return (trainset, testset), (trainloader, testloader)


def get_MNIST_dataset(dirpath="./data/MNIST"):
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


def load_MNIST_datasets_and_loaders(dirpath="./data/MNIST",
                                    batch_size=params.batch_size):
    """
    Returns:
        [Tuple]: 
            (trainset, testset), (trainloader, testloader)
    """

    trainset, testset = get_MNIST_dataset(dirpath)
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
    """
    teacher_loaders = []
    for i in range(data_size):
        indices = list(range(i*data_size, (i+1)*data_size))
        subset_data = Subset(data_loader, indices)
        loader = torch.utils.data.DataLoader(subset_data, batch_size=batch_size)
        teacher_loaders.append(loader)
    return teacher_loaders
    """


def dataset_from_indices(mask, dataset):
    indices = mask.nonzero()[0]
    if not (isinstance(indices, np.ndarray)):
        raise Exception(f"Indices are not numpy array: {type(indices)}")
    return Subset(dataset, indices)


def get_uneven_data_loaders(dataset, lengths):
    N = len(dataset)
    percent = int(N / 100)
    lengths = [
        percent * 30, percent * 20, percent * 15, percent * 10, percent * 5,
        percent * 5, percent * 5, percent * 5, percent * 5
    ]

    return torch.utils.data.random_split(dataset, lengths)
