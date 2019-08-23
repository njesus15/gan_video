from keras_preprocessing.image import save_img
from torch import optim
from torch.utils.data import DataLoader
from core.utils import *
from torchvision.transforms import ToPILImage
import logging
from core.model import *
from core.dataset import *
from examples.Logger import Logger
import numpy.random
import sys
import imageio
from skimage import img_as_ubyte


def train_gan(filename=None, epochs=100, l1_lambda=10):
    d_loss, g_loss, d_real_output, d_fake_output = 0, 0, 0, 0
    # Create logger
    logger = Logger(model_name='VGAN', data_name='bike')

    # Get dataloader
    vid_batch = 3
    data_loader = create_dataloader(filename=filename, video_batches=vid_batch)

    # Instantiate generator and discriminator
    generator = VideoGen().float()
    discriminator = VideoDiscriminator().float()

    # Directory to save generator videos
    DIR_TO_SAVE = "./gen_videos/"
    if not os.path.exists(DIR_TO_SAVE):
        os.makedirs(DIR_TO_SAVE)

    # Define optimizer for the discriminator and generator
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=5e-5)
    g_optimizer = optim.RMSprop(generator.parameters(), lr=5e-5)

    # first frames list
    first_frames = []

    for epoch in range(epochs):
        for (iter_num, batch) in enumerate(data_loader, 1):
            rand_int = numpy.random.randint(0, 32)
            if int(batch['video'].size(0) // 32) == vid_batch:
                real_video = batch['video'].view(vid_batch, 32, 3, 64, 64)
                filenames = batch['filename']
                real_randframe = real_video[:, rand_int:rand_int + 1, :, :, :]
                real_first_frame = real_video[:, 0:1 , :, :, :]

                if epoch == 0:
                    first_frames.append(real_first_frame)

                # Generate a fake video using first frame
                fake_video = generator(real_first_frame.squeeze())

                # reset gradients
                generator.zero_grad()
                discriminator.zero_grad()

                # Train discriminator
                if not iter_num % 2 == 0:
                    # detach fake video (generators parameters)
                    fake_video.detach()
                    d_loss, d_real_output, d_fake_output = d_loss_step(generator=generator,
                                                                       discriminator=discriminator,
                                                                       real_video=real_video,
                                                                       fake_video=fake_video)

                    # Update Gradient
                    d_loss.backward()
                    d_optimizer.step()

                    # Clamp
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                else:  # Train generator
                    g_loss, d_fake_output = g_loss_step(discriminator=discriminator,
                                                        fake_video=fake_video,
                                                        real_randframe=real_randframe,
                                                        real_first_frame = real_first_frame,
                                                        rand_int=rand_int,
                                                        l1_lambda=l1_lambda)

                    g_loss.backward()
                    g_optimizer.step()

                if iter_num % 2 == 0:
                    logger.display_status(
                        epoch, epochs, iter_num, len(data_loader),
                        d_loss, g_loss, d_real_output, d_fake_output
                    )

        real_frame_id = epoch % (len(data_loader) - 2)
        gen_out = generator(first_frames[real_frame_id].squeeze())

        t = gen_out.data.cpu()[0:1, :, 0:1, :, :].squeeze()

        save_img(x=np.transpose(t, (1, 2, 0)),
                 path=DIR_TO_SAVE + 'fake_img_sample_it%s_epoch%s.jpg' % (iter_num, epoch))

        # Save video from generator
        make_gif(denorm(gen_out.data.cpu()[0]),
                 DIR_TO_SAVE + 'fake_gifs_sample_it%s_epoch%s.gif' % (iter_num, epoch))

    return generator, discriminator


if __name__ == '__main__':
    gen, dis = train_gan('vid1.pickle', epochs=1, l1_lambda=50)
