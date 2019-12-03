import os
import numpy as np
# import cv2 as cv # use openCV if you want to generate the base activation image
import pickle
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset

import keras

def load_CIFAR10(root='././datasets/'):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root=root,
                                             train=True,
                                             download=False,
                                             transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root=root,
                                            train=False,
                                            download=False,
                                            transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=4,
                                              shuffle=False,
                                              num_workers=2)
    return train_loader, test_loader

def load_module(filename='models/cifar_10/trained_model/VGGNet_parameters.pkl'):
    module = torch.load(filename)
    return module

def get_activation(img='activation_base.jpg'):
    # using gradient and weights to calculate transformation on each channel
    pass

for i in range(10):
    # y = get_activation(img)
    # y.backward()
    pass


# class ActivationMaximization(Loss):
# """A loss function that maximizes the activation of a set of filters within a particular layer.
# Typically this loss is used to ask the reverse question - What kind of input image would increase the networks
# confidence, for say, dog class. This helps determine what the network might be internalizing as being the 'dog'
# image space.
# One might also use this to generate an input image that maximizes both 'dog' and 'human' outputs on the final
# `keras.layers.Dense` layer.
# """

#     def __init__(self, layer, filter_indices):
#         """
#         Args:
#             layer: The keras layer whose filters need to be maximized. This can either be a convolutional layer
#                 or a dense layer.
#             filter_indices: filter indices within the layer to be maximized.
#                 For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
#                 If you are optimizing final `keras.layers.Dense` layer to maximize class output, you tend to get
#                 better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
#                 output can be maximized by minimizing scores for other classes.
#         """
#         super(ActivationMaximization, self).__init__()
#         self.name = "ActivationMax Loss"
#         self.layer = layer
#         self.filter_indices = utils.listify(filter_indices)

#     def build_loss(self):
#         layer_output = self.layer.output

#         # For all other layers it is 4
#         is_dense = K.ndim(layer_output) == 2

#         loss = 0.
#         for idx in self.filter_indices:
#             if is_dense:
#                 loss += -K.mean(layer_output[:, idx])
#             else:
#                 # slicer is used to deal with `channels_first` or `channels_last` image data formats
#                 # without the ugly conditional statements.
#                 loss += -K.mean(layer_output[utils.slicer[:, idx, ...]])

#         return loss


# def get_img():
#     img = np.zeros([32, 32, 3], np.uint8)
#     for i in range(0,3):
#         img[:, :,i] = np.zeros([32, 32])+ 127
#     cv.imwrite('activation_base.jpg', img)

if __name__ == "__main__":
    # get_img()
    module = load_module()
    # print(module)
    get_activation()