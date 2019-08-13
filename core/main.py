from keras_preprocessing.image import save_img
from torch import optim
from torch.utils.data import DataLoader
from utils import *
from torchvision.transforms import ToPILImage
import logging
from core.model import *
from core.dataset import *
from examples.Logger import Logger
import sys
import imageio
from skimage import img_as_ubyte


def make_gif(images, filename):
    # Receives [3,32,64,64] tensor, and creates a gif

    x = images.permute(1,2,3,0)
    x = x.numpy()
    x = img_as_ubyte(x)
    frames = []
    for i in range(32):
        frames += [x[i]]
    imageio.mimsave(filename, frames, duration=0.1)

def denorm(x):
    out = (x + 1.0) / 2.0
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
    img = img.detach()
    pil_img = ToPILImage()
    img = pil_img(img)
    img.save(filename)


if __name__ == '__main__':
    # Instantiate logger parameters

    d_loss, g_loss, d_real_output, d_fake_output = 0, 0, 0, 0
    logger = Logger(model_name='VGAN', data_name='bike')
    l1_lambda = 10
    num_epochs = 120

    # Create binary cross entropy function
    loss = torch.nn.BCELoss()
    batch_size = 32*15

    # Define transforms to apply to the data
    composed = transforms.Compose([Rescale(64), ToTensor()])

    # Define location of data pickle files
    # Batches of 32 sequential frame paths are grouped together in a pandas df
    #working_dir = %pwd
    working_dir = '/Users/jesusnavarro/Desktop/gan_video'
    sub_dir = '/pickle_data/boat9_video_test_20_batches.pickle'

    # Instantiate DataLoader object
    transformed_uav_dataset = UAVDataset(pickle_path=working_dir + sub_dir,
                                         transform=composed)
    data_loader = DataLoader(transformed_uav_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    num_batches = len(data_loader)
    print(num_batches)

    # Instantiate generator and discriminator
    generator = VideoGen().float()
    discriminator = VideoDiscriminator().float()

    # Directory to save generator videos
    DIR_TO_SAVE = "./gen_videos/"
    if not os.path.exists(DIR_TO_SAVE):
        os.makedirs(DIR_TO_SAVE)
    sample_input = None
    sample_input_set = False

    loss_function = nn.CrossEntropyLoss()

    # Define optimizer for the discriminator and generator

    #d_optimizer = optim.RMSprop(discriminator.parameters(), lr=5e-5)
    #g_optimizer = optim.RMSprop(generator.parameters(), lr=5e-5)

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.555, 0.5))
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.555, 0.5))

    i = 0

    for epoch in range(num_epochs):
        save = True

        for (iter_num, batch) in enumerate(data_loader, 1):
            print(iter_num)
            if i == 0:
                real_video = batch['video'][0]
                real_video = real_video.view(int(real_video.size(0) // 32), 32, 3, 64, 64)
                filenames = batch['video'][1]
                #real_video = torch.squeeze(real_video, dim=1).unsqueeze(0) # (1,32,3,64,64)
                real_first_frame = real_video[:, 0:1, :, :, :]
                i += 1
            # reset gradients
            generator.zero_grad()
            discriminator.zero_grad()
            #real_video = batch['video']
            #real_video = torch.squeeze(real_video, dim=1)
            #real_video = real_video.permute(0, 2, 1, 3, 4)
            N = real_video.size(0)

            real_labels = torch.LongTensor(np.ones(N, dtype=int))
            fake_labels = torch.LongTensor(np.zeros(N,  dtype=int))


            # Train discriminator
            if not iter_num % 2 == 0:
                d_real_output_sig, d_real_output = discriminator(real_video.permute(0, 2, 1, 3, 4).float())
                d_real_output_sig, d_real_output = d_real_output_sig.squeeze(), d_real_output.squeeze()

                # Generate a fake video, detach parameters
                fake_video = generator(real_first_frame.squeeze()).detach()
                d_fake_output_sig, d_fake_output = discriminator(fake_video)

                # Compute real and fake loss
                if N == 1:
                    d_fake_output = d_fake_output.squeeze().unsqueeze(0)
                    d_real_output = d_real_output.squeeze().unsqueeze(0)
                else:
                    d_fake_output = d_fake_output.squeeze()
                    d_real_output = d_real_output.squeeze()

                #d_loss = -(torch.mean(d_real_output) - torch.mean(d_fake_output))
                d_loss = loss(d_real_output_sig.squeeze().float(), real_labels.float()) + \
                         loss(d_fake_output_sig.squeeze().float(), fake_labels.float())


                # Update Gradient
                d_loss.backward()
                d_optimizer.step()

                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            else:  # Train generator
                real_first_frame = real_video[:, 0:1, :, :, :]
                fake_videos = generator(real_first_frame.squeeze())
                d_fake_outputs_sig, d_fake_outputs = discriminator(fake_videos)
                d_fake_outputs_sig, d_fake_outputs = d_fake_outputs_sig.squeeze(), d_fake_outputs.squeeze()
                fake_first_frame = fake_videos[:, :, 0:1, :, :]
                reg_loss = torch.mean(torch.abs(real_first_frame.float() - fake_first_frame.float())) * l1_lambda
                #g_loss = -torch.mean(d_fake_outputs) + reg_loss
                g_loss = loss(d_fake_outputs_sig.float(), real_labels.float())
                g_loss.backward()
                g_optimizer.step()

            if iter_num % 2 == 0:
                logger.display_status(
                    epoch, num_epochs, iter_num, len(data_loader),
                    d_loss, g_loss, d_real_output, d_fake_output
                )

                if (epoch + 1) % 3 == 0 and save:

                    gen_out = generator(real_first_frame)

                    t = gen_out.data.cpu()[0:1, :, 0:1, :, :].squeeze()

                    save_img(x=np.transpose(t, (1, 2, 0)),
                             path=DIR_TO_SAVE + 'fake_img_sample_it%s_epoch%s.jpg' % (iter_num, epoch))

                    # Save video from generator
                    make_gif(denorm(gen_out.data.cpu()[0]),
                             DIR_TO_SAVE + 'fake_gifs_sample_it%s_epoch%s.gif' % (iter_num, epoch))

                    save = False





