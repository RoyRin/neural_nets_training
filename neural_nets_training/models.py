import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_nets_training import params
import toolz
import numpy as np


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


"""
TODO:
1. make it possible to use different models on different datasets
2. recreate this fast resnet 9    # https://dawn.cs.stanford.edu/benchmark/CIFAR10/train.html for CIFAR
"""


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.basemodel_name = "ResNet9"
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        # yapf: disable
        return toolz.pipe(xb,   self.conv1,
                                self.conv2,
                                lambda x: self.res1(x) + x,
                                self.conv3,
                                self.conv4,
                                lambda x: self.res2(x) + x,
                                self.classifier)
        # yapf: enable


class FCN(nn.Module):
    def __init__(self, num_classes=10, input_pixels=28 * 28):
        super(FCN, self).__init__()
        self.basemodel_name = "FCN"
        self.input_pixels = input_pixels
        self.fc1 = nn.Linear(self.input_pixels, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_pixels)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class memorization_FCN(nn.Module):
    def __init__(self, num_classes=10, input_pixels=28 * 28):
        """
        NN attempting to mimic Jax implementation - 
        https://github.com/google-research/heldout-influence-estimation/blob/master/mnist-example/mnist_infl_mem.py 
        init_random_params, predict = stax.serial(Flatten,
                                                Dense(512), Relu, 
                                                Dense(256), Relu,
                                                Dense(10), LogSoftmax)

        """
        super(memorization_FCN, self).__init__()
        self.basemodel_name = "memorization_FCN"
        self.input_pixels = input_pixels
        self.fc1 = nn.Linear(input_pixels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_pixels)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


# CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()  # super(Net, self).__init__()
        self.basemodel_name = "CNN"
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # x represents our data
    def forward(self, x):
        # yapf: disable
        return toolz.pipe(x, self.conv1, F.relu,
                            self.conv2, F.relu,
                            lambda x: F.max_pool2d(x, 2),
                            self.dropout1,
                            lambda x: torch.flatten(x, 1),
                            self.fc1, F.relu,
                            self.dropout2,
                            self.fc2,
                            lambda x: F.log_softmax(x, dim=1))
        # yapf: enable


def parameters_per_model(model):
    return sum([np.prod(i.shape) for i in model.parameters()])


def CNN_model_factory(*, num_classes=10, device=params.get_default_device()):
    """Helper, returns a CNN model"""
    model = CNN(num_classes=num_classes)
    model.to(device)
    return model


def FCN_model_factory(*, num_classes=10, device=params.get_default_device()):
    """Helper, returns a FCN model"""
    model = FCN(num_classes=num_classes)
    model.to(device)
    return model


def Resnet9_model_factory(*,
                          in_channels=3,
                          num_classes=10,
                          device=params.get_default_device()):
    """Helper, returns a ResNet9 model"""
    model = ResNet9(in_channels=in_channels, num_classes=num_classes)
    model.to(device)
    return model
