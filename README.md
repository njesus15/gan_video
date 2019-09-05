# Video GAN - Predicting Future Frames from Single Images
The goal of Video GAN is to train a generative advarsarial network to generate short synthetic videos from input images. In this specific application, VideoGAN is trained on close-up shots of basketball highlights, and thus the objective for the network is to learn the video dynamics displayed by the players.

## Network architecture

![alt text](http://www.cs.columbia.edu/~vondrick/tinyvideo/network.png)

The network of the achitecture, as shown above, is implemented from [Generating Videos with Scene Dynamics](http://www.cs.columbia.edu/~vondrick/tinyvideo/) by Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. The general apporoach is to use two streams foreground and background, which learn the dynamics of the videos and static background, respectively. The discriminator decided whether the input video is real or synthetic. It's goal is to correctly differentiate real from synthetic, while the generator's goal to fool the discriminator.

## Training
The GAN was trained on a local machine and thus a relatively small size of data was used for training. Training was done with a batch size of 3 videos which amounts to 96 images and 120 epochs. It took several hours to train the data at its original fps. Various experiments were performed in which the video frames were extracted at different fps with 32 frames per video. Additionally, for every third step, the generator was updated to account for a larger amount of training on the generator.

## Data
Video data was web-scraped from YouTube. See 'core/preprocessed.py', which provides a method to extract a certain number of videos requested and specify the maximum duration of the video in seconds. 
#### NOTE: Current bug in preprocessed.py with PyTube library needed to fix at the moment.

## Code
The core directory contains the implementation of Video GAN:

#### dataset.py
- Inherits PyTorch's DataSet class to construct a dataloader

#### model.py
- PyTorch implementation of the neural network architecture

#### preprocess.py
- Script to web-scrape video data from YouTube

#### main.py
- Trains the neural network 

To train, create a virtual environment and install requirements.txt and run main.py.

## Results/Current State
Given compute limitations on my local machine, the size of the generated videos are small (32, 3, 64, 64). So synthetic videos are very distinguishable from real videos. However, results show that Video GAN is able to learn dynamics and predict reasonable spatial predictions. 

I am still working on optimizing the training of the Video GAN to extract optmimal results. In addition, I am working on developing a web-based API of Video GAN.
