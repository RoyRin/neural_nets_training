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
    """ Resnet 9 """
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
    """ 2 layer fully connected network """
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
    """ 2 layer fully connected network  which was used in memorization paper 
    https://github.com/google-research/heldout-influence-estimation/blob/master/mnist-example/mnist_infl_mem.py  """
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
    """2 layer CNN"""
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
        """forward inference"""
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


class cifar10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(cifar10_CNN, self).__init__()
        self.basemodel_name = "cifar10_CNN"
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# code taken from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))

    def forward(self, x):
        """forward inference"""
        return nn.ReLU(inplace=True)(self.residual_function(x) +
                                     self.shortcut(x))


class ResNet(nn.Module):
    """ Large Resnet base class"""
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        """ forward inference for resnet"""
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def parameters_per_model(model):
    return sum([np.prod(i.shape) for i in model.parameters()])


def CNN_model_factory(*,
                      num_classes=10,
                      device=params.get_default_device(),
                      seed=None):
    """Helper, returns a CNN model"""
    if seed is not None:
        torch.manual_seed(seed)
    model = CNN(num_classes=num_classes)
    model.to(device)
    return model


def cifar10_CNN_model_factory(*,
                              num_classes=10,
                              device=params.get_default_device(),
                              seed=None):
    """Helper, returns a cifar_CNN model"""
    if seed is not None:
        torch.manual_seed(seed)
    model = cifar10_CNN(num_classes=num_classes)
    model = model.to(device)
    return model


def FCN_model_factory(*,
                      num_classes=10,
                      device=params.get_default_device(),
                      seed=None):
    """Helper, returns a FCN model"""
    if seed is not None:
        torch.manual_seed(seed)
    model = FCN(num_classes=num_classes)
    model.to(device)
    return model


def Resnet9_model_factory(*,
                          in_channels=3,
                          num_classes=10,
                          device=params.get_default_device(),
                          seed=None):
    """Helper, returns a ResNet9 model"""
    if seed is not None:
        torch.manual_seed(seed)
    model = ResNet9(in_channels=in_channels, num_classes=num_classes)
    model.to(device)
    return model


def Resnet50_model_factory(*,
                           num_classes=100,
                           device=params.get_default_device(),
                           seed=None):
    """ return a ResNet 50 object
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)
    model.to(device)
    return model


def Resnet101_model_factory(*,
                            num_classes=10,
                            device=params.get_default_device(),
                            seed=None):
    """ return a ResNet 101 object """
    if seed is not None:
        torch.manual_seed(seed)
    model = ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)
    model.to(device)
    return model


def Resnet152_model_factory(*,
                            num_classes=10,
                            device=params.get_default_device(),
                            seed=None):
    """ return a ResNet 152 object """
    if seed is not None:
        torch.manual_seed(seed)
    model = ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)
    model.to(device)
    return model
