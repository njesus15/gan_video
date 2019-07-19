import os
import numpy as np
import errno
import torchvision.utils as vutils
from imageio import imread
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
from tensorflow.python import Variable
import tensorflow as tf


class VideoDiscriminator(nn.Module):

    def __init__(self):
        super(VideoDiscriminator, self).__init__()  # input: (-1, 3, 32, 64, 64)
        self.hidden0 = nn.Sequential(nn.Conv3d(in_channels=3,
                                               out_channels=64,
                                               kernel_size=2,
                                               stride=2),
                                     nn.LeakyReLU(0.2))

        self.hidden1 = nn.Sequential(nn.Conv3d(in_channels=64,
                                               out_channels=128,
                                               kernel_size=2,
                                               stride=2),
                                     nn.BatchNorm3d(128),
                                     nn.LeakyReLU(0.2))

        self.hidden2 = nn.Sequential(nn.Conv3d(in_channels=128,
                                               out_channels=256,
                                               kernel_size=2,
                                               stride=2),
                                     nn.BatchNorm3d(256),
                                     nn.LeakyReLU(0.2))

        self.hidden3 = nn.Sequential(nn.Conv3d(in_channels=256,
                                               out_channels=512,
                                               kernel_size=2,
                                               stride=2),
                                     nn.BatchNorm3d(512),
                                     nn.LeakyReLU(0.2))

        self.hidden4 = nn.Sequential(nn.Conv3d(in_channels=512,
                                               out_channels=2,
                                               kernel_size=(2, 4, 4),
                                               stride=(1, 1, 1)
                                               )
                                     )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)


        return x


class ForegroundStream(nn.Module):
    """ Generative CNN Mode

    Defualt input image size = (101, 101, 3)
    Output image size =  (64, 64, 3)
    """

    def __init__(self):
        super(ForegroundStream, self).__init__()
        self.hidden0 = nn.Sequential(nn.ConvTranspose3d(in_channels=1,
                                                        out_channels=512,
                                                        kernel_size=(2, 4, 4),
                                                        stride=(1, 1, 2),
                                                        padding=(0, 0, 99),
                                                        output_padding=0),
                                     nn.BatchNorm3d(512),
                                     nn.ReLU(True))

        self.hidden1 = nn.Sequential(nn.ConvTranspose3d(in_channels=512,
                                                        out_channels=256,
                                                        kernel_size=(4, 4, 4),
                                                        padding=1,
                                                        stride=2),
                                     nn.BatchNorm3d(256),
                                     nn.ReLU(True)
                                     )
        self.hidden2 = nn.Sequential(nn.ConvTranspose3d(in_channels=256,
                                                        out_channels=128,
                                                        kernel_size=(4, 4, 4),
                                                        padding=1,
                                                        stride=2),
                                     nn.BatchNorm3d(128),
                                     nn.ReLU(True)
                                     )
        self.hidden3 = nn.Sequential(nn.ConvTranspose3d(in_channels=128,
                                                        out_channels=64,
                                                        kernel_size=(4, 4, 4),
                                                        padding=1,
                                                        stride=2),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(True)
                                     )
        self.hidden4 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,
                                                        out_channels=3,
                                                        kernel_size=(4, 4, 4),
                                                        padding=1,
                                                        stride=2),
                                     nn.Tanh()
                                     )
        self.mask = nn.Sequential(nn.ConvTranspose3d(in_channels=64,
                                                     out_channels=1,
                                                     kernel_size=(4, 4, 4),
                                                     padding=1,
                                                     stride=2),
                                  nn.Sigmoid()
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
        mask = self.mask(x)
        gen = self.hidden4(x)

        return gen, mask


class BackgroundStream(nn.Module):

    def __init__(self):
        super(BackgroundStream, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1,
                               out_channels=512,
                               kernel_size=4,
                               stride=(1, 2),
                               padding=(0, 99)
                               ),
            nn.BatchNorm2d(512),
            nn.ReLU(True))

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=4,
                               stride=(2, 2),
                               padding=(1, 1)
                               ),
            nn.BatchNorm2d(256),
            nn.ReLU(True))

        self.hidden2 = nn.Sequential(nn.ConvTranspose2d(in_channels=256,
                                                        out_channels=128,
                                                        kernel_size=4,
                                                        stride=(2, 2),
                                                        padding=(1, 1)
                                                        ),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.hidden3 = nn.Sequential(nn.ConvTranspose2d(in_channels=128,
                                                        out_channels=64,
                                                        kernel_size=4,
                                                        stride=(2, 2),
                                                        padding=(1, 1)
                                                        ),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True))
        self.hidden4 = nn.Sequential(nn.ConvTranspose2d(in_channels=64,
                                                        out_channels=3,
                                                        kernel_size=4,
                                                        stride=(2, 2),
                                                        padding=(1, 1)
                                                        ),
                                     nn.Tanh()
                                     )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)

        return x


class VideoGen(nn.Module):

    def __init__(self):
        super(VideoGen, self).__init__()
        self.fg_stream = ForegroundStream()
        self.bg_stream = BackgroundStream()

    def forward(self, x):
        assert np.ndim(x.detach().numpy()) == 5

        fg, mask = self.fg_stream(x)  # (-1, 3, 32, 64, 64), (-1, 1, 32, 64, 64),
        background = self.bg_stream(x[0])  # (-1, 3, 64, 64)
        background_frames = background.unsqueeze(2).repeat(1, 1, 32, 1, 1)
        mask_frames = mask.repeat(1, 3, 1, 1, 1)

        # out: torch.matmul(mask, fg) + np.matmul((1 - mask), background)
        out = torch.mul(mask_frames, fg) + torch.mul((1 - mask), background_frames)

        return out
