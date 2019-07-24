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


class DiscrminativeModel(torch.nn.Module):
    """ Discriminative CNN Model for WGAN which does not output a probability and
        hence no sigmoid activation

    Default image input size = (101, 101, 3)

    """

    def __init__(self, image_size=101):
        super(DiscrminativeModel, self).__init__()
        self.hidden0 = torch.nn.Sequential(torch.nn.Linear(12288, 1000),
                                           torch.nn.ReLU())
        self.hidden1 = torch.nn.Sequential(torch.nn.Linear(1000, 500),
                                           torch.nn.ReLU())
        self.hidden5 = torch.nn.Sequential(torch.nn.Linear(500, 200),
                                           torch.nn.ReLU())
        self.hidden6 = torch.nn.Sequential(torch.nn.Linear(200, 1),
                                           )

    def forward(self, x):
        """
        :param x: input array of img
        :return: (numpy.ndarray) outer layer of cnn
        """
        x = x.view(x.size(0), -1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden5(x)
        x = self.hidden6(x)


        return x

    def train_disriminator(self, optimizer, real_imgs, fake_imgs):
        # initialize weights
        optimizer.zero_grad()

        # Compute error
        disc = back_propagate(self, real_imgs, fake_imgs)

        # propagate error
        disc[0].backward()
        optimizer.step()

        return disc

def back_propagate(discrminator, real_images, fake_images):

    real_predicition = discrminator(real_images.float())
    fake_prediction = discrminator(fake_images.float())

    d_fake = torch.mean(fake_prediction)
    d_real = torch.mean(real_predicition)

    disc_loss = d_fake - d_real

    return disc_loss, real_predicition, fake_prediction


