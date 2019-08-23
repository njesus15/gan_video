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

        :param csv_file_path: path to saved csv file
        :param root_dir: root directory where data is stored
        :param transform: tensor.transforms applied to data
        """
        with open(pickle_path, 'rb') as handle:
            self.flight_frames = pickle.load(handle)
        self.transform = transform

    def __len__(self):
        return len(self.flight_frames)

    def __getitem__(self, item_id):
        image_paths = self.flight_frames.iloc[item_id]['video']
        class_image_paths = self.flight_frames.iloc[item_id]['class']
        data = {'video': image_paths, 'class': class_image_paths, 'filename': image_paths}

        if self.transform:
            data = self.transform(data)

        # Apply transform to each image, then group into batches of 64!

        return data



class Rescale(object):
    """ Transform to rescale image
    """
    def __init__(self, output_size):
        """
        :param output_size: int to resize image (3 x output_size x output_size)
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        returns dictionary with key 'video' and img array and path

        """
        filename = sample['video']
        img = image.load_img(filename, target_size=(self.output_size, self.output_size))
        sample['video'] = img

        return sample


class ToTensor(object):
    """
    Converts data to tensors and applies normalization transform

    Input tensors are of size (N, Channels, Depth, Height, Width)

    TODO: Fix class values. """
    def __call__(self, sample):
        img = sample['video']
        trans = transforms.ToTensor()
        video = trans(img)
        norm_t = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        video = norm_t(video.float())
        video = video.unsqueeze(0)
        sample['video'] = video
        return sample

class Normalize():

    def __call__(self, sample):
        video = sample['video'][0]
        filename = sample['video'][1]
        x = transforms.Normalize(0.5, 0.5)
        video = x(video)
        return {'video': [video, filename]}
