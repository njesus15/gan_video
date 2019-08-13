from torchvision.transforms import ToPILImage
import logging
from core.model import *
import sys
import imageio
from skimage import img_as_ubyte


def make_gif(images, filename):
    # Receives [3,32,64,64] tensor, and saves in gif format

    x = images.permute(1, 2, 3, 0)
    x = x.numpy()
    x = img_as_ubyte(x)
    frames = []
    for i in range(32):
        frames += [x[i]]
    imageio.mimsave(filename, frames, duration=0.1)


def denorm(image):
    # denormalize tensor image
    out = (image + 1.0) / 2.0
    tf = nn.Tanh()
    return tf(out)


def setup_logger(logger_name):
    # format string
    LOG_FORMAT = "%(levelname)s - %(asctime)s - %(messages)s"
    logger = logging.getLogger(logger_name)

    # create formatter
    formatter = logging.Formatter(fmt=LOG_FORMAT)

    # create handler, streamhandler, and format
    file_handler = logging.FileHandler('training_log.txt', mode='w')
    file_handler.setFormatter(file_handler)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(screen_handler)

    return logger


def save_image(img, filename):
    # Saves PIL images to filename
    img = img.detach()
    pil_img = ToPILImage()
    img = pil_img(img)
    img.save(filename)
