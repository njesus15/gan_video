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
        print(self.flight_frames.head())
        self.transform = transform

    def __len__(self):
        return len(self.flight_frames)

    def __getitem__(self, item_id):
        video_frames = self.flight_frames.iloc[item_id]['video']
        data = {'video': video_frames}

        if self.transform:
            data = self.transform(data)
        return data


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample_video):
        video = sample_video['video']
        video_array = []
        for frame in video: # (path, class)
            img = image.load_img(frame[0], target_size=(self.output_size, self.output_size))
            img = np.array(img)
            video_array.append(img)
        return {'video': np.array(video_array)}


class ToTensor(object):
    """
    Converts data to tensors

    Input tensors are of size (N, Channels, Depth, Height, Width)
    TODO: Fix class values. """
    def __call__(self, sample):
        video = sample['video']
        video = video.transpose((3, 0, 1, 2)) #
        video = torch.from_numpy(video)
        video_dim5 = video.unsqueeze(0)
        return {'video': video_dim5}

class Normalize():

    def __call__(self, sample):
        video = sample['video']
        x = transforms.Normalize(0.5, 0.5)
        video = x(video)
        return {'video': video}
