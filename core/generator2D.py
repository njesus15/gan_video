import os
import numpy as np
import errno
import torchvision.utils as vutils
from imageio import imread
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch.nn
from tensorflow.python import Variable
import tensorflow as tf



class GenerativeModel(torch.nn.Module):
    """ Generative CNN Mode

    Defualt input image size = (101, 101, 3)
    Output image size =  (64, 64, 3)
    """

    def __init__(self, input_size=100, image_size=101):
        super(GenerativeModel, self).__init__()
        self.hidden0 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_size, 512, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True)
        )
        self.hidden1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True)
        )

        self.hidden2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True)
        )

        self.hidden3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True)
        )
        self.hidden4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )


    def forward(self, x):
        """
        :param x: input array of img
        :return: (numpy.ndarray) outer layer of cnn

        """
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)

        return x

    def train_generator(self, discriminator, optimizer, fake_imgs, loss_fcn):
        N = fake_imgs.size(0)
        optimizer.zero_grad()
        d_fake_predictions = discriminator(fake_imgs)
        error = -torch.mean(d_fake_predictions)
        error.backward()
        optimizer.step()

        return error

def gen_noise(size, length=200):
    """ Generates random tensor for generator"""
    n = torch.randn(size, length, 1, 1)
    return n
