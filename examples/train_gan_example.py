from PIL import Image
from torch import optim
from core.examples.Logger import Logger
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from core.gen_net import *
from core.disc_net import *
import matplotlib.pyplot as plt
from core.UAVDataset import *
from torch.autograd import Variable
import numpy as np


def vectors_to_images(vectors, image_size=64):
    return vectors.view(vectors.size(0), 3, image_size, image_size)

def train_gan(epochs=100, learning_rate = 0.001):
    logger = Logger(model_name='VGAN', data_name='bike')
    loss = torch.nn.BCELoss()
    img_size = 64
    input_size = 100
    num_test_samples = 16
    test_noise = gen_noise(num_test_samples, length=input_size)

    composed = transforms.Compose([Rescale(img_size), ToTensor()])
    transformed_uav_dataset = UAVDataset(pickle_path='/Users/jesusnavarro/Desktop/gan_video/core/data/test_200_images.pickle',
                                         transform=composed)
    data_loader = DataLoader(transformed_uav_dataset,
                             batch_size=10,
                             shuffle=True)
    num_batches = len(data_loader)

    generator = GenerativeModel(input_size=input_size, image_size=img_size).float()
    discriminator = DiscrminativeModel(image_size=img_size).float()

    # Instantiate optimizers
    dis_optimizer = optim.RMSprop(discriminator.parameters(), lr=5e-5)
    gen_optimizer = optim.RMSprop(generator.parameters(), lr=5e-5)

    for epoch_num in range(epochs):
        for batch_num, (real_batch) in enumerate(data_loader):
            data_iter = iter(data_loader)
            inputs, labels = real_batch['image'], real_batch['class']
            N = inputs.shape[0]

            # TRAIN DISCRIMINATOR (done more times than generator)
            for i in range(5):
                data = data_iter.next()
                inputs = data['image']

                # generate fake data and get prediction from generator
                z = gen_noise(N, length=input_size)

                # compute gen output, detach to optimize seperately
                fake_images = generator(z).detach()

                # convert input images to floats
                inputs.float()
                d_logs = discriminator.train_disriminator(dis_optimizer, inputs, fake_images)

                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # TRAIN GENERATOR
            generator_prediction = generator(gen_noise(N, length=input_size))
            gen_error = generator.train_generator(discriminator, gen_optimizer, generator_prediction, loss)

            logger.log(d_logs[0], gen_error, epoch_num, batch_num, num_batches)
            # Display Progress every few batches
            if (batch_num) % 5 == 0:
                test_images = vectors_to_images(generator(test_noise * 255), image_size=img_size)
                test_images = test_images.data
                logger.log_images(
                    test_images, num_test_samples,
                    epoch_num, batch_num, num_batches
                )
                # Display status Logs
                logger.display_status(
                    epoch_num, epochs, batch_num, num_batches,
                    d_logs[0], gen_error, d_logs[1], d_logs[2]
                )

    return generator, discriminator

if __name__ == '__main__':
    models = train_gan(learning_rate=0.0002)
