from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import logging
from core.model import *
import sys
import imageio
from skimage import img_as_ubyte

from core.utils import *
from torchvision.transforms import ToPILImage
import logging
from core.model import *
from core.dataset import *
import numpy as np
import cv2


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


def create_dataloader(filename, video_batches=2, shuffle=False):
    """ Helper function to create and return data loader
     """
    batch_size = 32 * video_batches

    # Define transforms to apply to the data
    composed = transforms.Compose([Rescale(64), ToTensor()])

    working_dir = '../pickle_data/'
    full_path = working_dir + filename

    try:
        os.path.exists(full_path)
    except:
        ValueError('filename -' + filename + '- not found. Make sure file exists within ./pickle_data/ directory')

    # Instantiate DataLoader object
    transformed_uav_dataset = UAVDataset(pickle_path=full_path,
                                         transform=composed)

    data_loader = DataLoader(transformed_uav_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader


def d_loss_step(discriminator, generator, real_video, fake_video):
    """
    Computes the discriminator loss

    :param discriminator: torch.nn.sequential
    :param generator: torch.nn.sequential
    :param real_video: tensor containing real frames (batch_num, 3, 32, 64, 64)
    :param fake_video: tensor containing generated frames (batch_num, 32, 3, 64, 64)

    :return: float loss

    """
    d_real_output_sig, d_real_output = discriminator(real_video.permute(0, 2, 1, 3, 4).float())
    d_real_output_sig, d_real_output = d_real_output_sig.squeeze(), d_real_output.squeeze()

    d_fake_output_sig, d_fake_output = discriminator(fake_video)
    d_fake_output_sig, d_fake_output = d_fake_output_sig.squeeze(), d_fake_output.squeeze()

    d_loss = -(torch.mean(d_real_output) - torch.mean(d_fake_output))

    return d_loss, d_real_output, d_fake_output


def g_loss_step(discriminator, fake_video, real_randframe, real_first_frame, rand_int, l1_lambda):
    """ TODO: Double check the subtraction from fake and random frame """
    d_fake_outputs_sig, d_fake_outputs = discriminator(fake_video)
    d_fake_outputs_sig, d_fake_outputs = d_fake_outputs_sig.squeeze(), d_fake_outputs.squeeze()
    fake_randframe = fake_video[:, :, rand_int:rand_int + 1, :, :].permute(0, 2, 1, 3, 4)
    frame_pred_loss = float(rand_int) / 32.0 * torch.mean(torch.abs(real_first_frame.float() - real_randframe.float()))
    reg_loss = torch.mean(torch.abs(real_randframe.float() - fake_randframe.float())) * l1_lambda
    g_loss = -torch.mean(d_fake_outputs) + reg_loss - frame_pred_loss

    return g_loss, d_fake_outputs


def extract_frames(filename):
    """ Saves video into frames and returns file paths"""
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


