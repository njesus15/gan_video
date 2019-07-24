from PIL import Image
from imageio import imread
from torchvision import transforms, utils
import os
import pickle
import numpy as np
from keras.preprocessing import image
from skimage import io, transform

import torch
import pandas as pd


class UAVDataset(torch.utils.data.Dataset):
    """ UAV Flight DataSet
    Inherits DataSet abstract class from pytroch
    Overrides methods __len__ and __getitem__
    """

    def __init__(self, pickle_path, transform=None):
        """

        :param csv_file_path:
        :param root_dir:
        :param transform:
        """
        with open(pickle_path, 'rb') as handle:
            self.flight_frames = pickle.load(handle)
        self.transform = transform

    def __len__(self):
        return len(self.flight_frames)

    def __getitem__(self, item_id):
        video_frames = self.flight_frames.iloc[item_id]['video']
        data = {'video': video_frames}

        if self.transform:
            data = self.transform(data)

        # Apply transform to each image, then group into batches of 64!

        return data



class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        filename = sample['video']
        img = image.load_img(filename, target_size=(self.output_size, self.output_size))
        img = np.array(img) / 255.0
        return {'video': [img, filename]}


class ToTensor(object):
    """
    Converts data to tensors

    Input tensors are of size (N, Channels, Depth, Height, Width)
    TODO: Fix class values. """
    def __call__(self, sample):
        img = sample['video'][0]
        filename = sample['video'][1]
        trans = transforms.ToTensor()
        video = trans(img)
        #video = video.permute(1, 2, 0)
        norm_t = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        video = norm_t(video.float())
        video = video.unsqueeze(0)
        return {'video': [video, filename]}

class Normalize():

    def __call__(self, sample):
        video = sample['video'][0]
        filename = sample['video'][1]
        x = transforms.Normalize(0.5, 0.5)
        video = x(video)
        return {'video': [video, filename]}
