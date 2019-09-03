import torch
import torch.nn as nn


class Encoder(nn.Module):
    # Image encoder into latent space with dim 100
    # TODO: Modify architecture of encoder network  to increase performance

    def __init__(self):
        super(Encoder, self).__init__()

        self.hidden0 = nn.Sequential(nn.Conv2d(in_channels=3,
                                               out_channels=64,
                                               kernel_size=(2, 2),
                                               stride=(2, 2),
                                               padding=(0, 0)
                                               ),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True))

        self.hidden1 = nn.Sequential(nn.Conv2d(in_channels=64,
                                               out_channels=128,
                                               kernel_size=(2, 2),
                                               stride=(2, 2),
                                               ),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))

        self.hidden2 = nn.Sequential(nn.Conv2d(in_channels=128,
                                               out_channels=256,
                                               kernel_size=(2, 2),
                                               stride=(2, 2),
                                               ),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(True))
        self.hidden3 = nn.Sequential(nn.Conv2d(in_channels=256,
                                               out_channels=512,
                                               kernel_size=(2, 2),
                                               stride=(2, 2),
                                               ),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(True))

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)

        return x


class VideoDiscriminator(nn.Module):

    # Discrimiator network with 32 frames and  input size of 32, 64, 64
    def __init__(self):
        super(VideoDiscriminator, self).__init__()  # input: (-1, 3, 32, 64, 64)
        self.hidden0 = nn.Sequential(nn.Conv3d(in_channels=3,
                                               out_channels=64,
                                               kernel_size=(4, 4, 4),
                                               padding=1,
                                               stride=(2, 2, 2)),
                                     nn.LeakyReLU(0.2))

        self.hidden1 = nn.Sequential(nn.Conv3d(in_channels=64,
                                               out_channels=128,
                                               kernel_size=(4, 4, 4),
                                               stride=(2, 2, 2),
                                               padding=1),
                                     nn.BatchNorm3d(128),
                                     nn.LeakyReLU(0.2))

        self.hidden2 = nn.Sequential(nn.Conv3d(in_channels=128,
                                               out_channels=256,
                                               kernel_size=(4, 4, 4),
                                               stride=(2, 2, 2),
                                               padding=1),
                                     nn.BatchNorm3d(256),
                                     nn.LeakyReLU(0.2))

        self.hidden3 = nn.Sequential(nn.Conv3d(in_channels=256,
                                               out_channels=512,
                                               kernel_size=(4, 4, 4),
                                               stride=(2, 2, 2),
                                               padding=1),
                                     nn.BatchNorm3d(512),
                                     nn.LeakyReLU(0.2))

        self.hidden4 = nn.Sequential(nn.Conv3d(in_channels=512,
                                               out_channels=1,
                                               kernel_size=(2, 4, 4),
                                               stride=(1, 1, 1),
                                               padding=0
                                               )
                                     )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        sig = torch.nn.Sigmoid()

        return sig(x), x


class ForegroundStream(nn.Module):
    """ Generative CNN Model

    Defualt input image size = (101, 101, 3)
    Output image size =  (64, 64, 3)
    """

    def __init__(self):
        super(ForegroundStream, self).__init__()


        self.hidden1 = nn.Sequential(nn.ConvTranspose3d(in_channels=512,
                                                        out_channels=256,
                                                        kernel_size=(4, 4, 4),
                                                        padding=1,
                                                        stride=(2, 2, 2)),
                                     nn.BatchNorm3d(256),
                                     nn.ReLU(True)
                                     )
        self.hidden2 = nn.Sequential(nn.ConvTranspose3d(in_channels=256,
                                                        out_channels=128,
                                                        kernel_size=(4, 4, 4),
                                                        padding=1,
                                                        stride=(2, 2, 2)),
                                     nn.BatchNorm3d(128),
                                     nn.ReLU(True)
                                     )
        self.hidden3 = nn.Sequential(nn.ConvTranspose3d(in_channels=128,
                                                        out_channels=64,
                                                        kernel_size=(4, 4, 4),
                                                        padding=1,
                                                        stride=(2, 2, 2)),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(True)
                                     )

    def forward(self, x):
        """
        :param x: input array of img
        :return: (numpy.ndarray) outer layer of cnn

        """
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)

        return x


class BackgroundStream(nn.Module):

    def __init__(self):
        super(BackgroundStream, self).__init__()


        self.hidden1 = nn.Sequential(nn.ConvTranspose2d(in_channels=512,
                                                        out_channels=256,
                                                        kernel_size=(2, 2),
                                                        stride=(2, 2),
                                                        ),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(True))

        self.hidden2 = nn.Sequential(nn.ConvTranspose2d(in_channels=256,
                                                        out_channels=128,
                                                        kernel_size=(2, 2),
                                                        stride=(2, 2),
                                                        ),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.hidden3 = nn.Sequential(nn.ConvTranspose2d(in_channels=128,
                                                        out_channels=64,
                                                        kernel_size=(2, 2),
                                                        stride=(2, 2),
                                                        ),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True))
        self.hidden4 = nn.Sequential(nn.ConvTranspose2d(in_channels=64,
                                                        out_channels=3,
                                                        kernel_size=(2, 2),
                                                        stride=(2, 2),
                                                        ),
                                     nn.Tanh()
                                     )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)

        return x


class VideoGen(nn.Module):

    # Network for video generation of 32 frames of dim 3, 64, 64
    # Input size dim is 100

    def __init__(self):
        super(VideoGen, self).__init__()
        self.encode = Encoder()
        self.fg_stream = ForegroundStream()

        self.video = nn.Sequential(nn.ConvTranspose3d(in_channels=64,
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
                                                     stride=(2, 2, 2)),
                                  nn.Sigmoid()
                                  )
        self.bg_stream = BackgroundStream()

    def forward(self, x):
        latent = self.encode(x).unsqueeze(dim=2)
        video = self.fg_stream(latent.repeat(1, 1, 2, 1, 1))
        fg = self.video(video)
        mask = self.mask(video)
        background = self.bg_stream(latent.squeeze(dim=2))  # (-1, 3, 64, 64)
        background_frames = background.unsqueeze(2).repeat(1, 1, 32, 1, 1)
        mask_frames = mask.repeat(1, 3, 1, 1, 1)

        # out: torch.matmul(mask, fg) + np.matmul((1 - mask), background)
        out = torch.mul(mask_frames, fg) + torch.mul((1 - mask), background_frames)

        return out
