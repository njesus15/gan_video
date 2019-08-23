import pwd

import pytube
import os
from core.utils import *
from bs4 import BeautifulSoup as bs
import requests
import cv2

import random


def save():
    vid_dir = '../mp4videos/'
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    url = 'https://www.youtube.com/watch?v=zGLZxZmLNo8'
    yt = pytube.YouTube(url)
    stream = yt.streams.first()
    stream.download('../mp4videos/')


def get_youtube_links(keywords=None):
    """
    Extract youtube likes based on keywords

    :param keywords: str appended to search query

    :return: dict with keyword keys and video links as values

    """

    query_link = "https://www.youtube.com/results?search_query=basketball+highlights+"
    video_links = {}

    # get links for each search

    for key in keywords:
        r = requests.get(query_link + key)
        page = r.text
        soup = bs(page, 'html.parser')
        vids = soup.findAll('a', attrs={'class': 'yt-uix-tile-link'})
        tmp_vidlist = []

        for v in vids:
            tmp = 'https://www.youtube.com' + v['href']
            tmp_vidlist.append(tmp)

        video_links[key] = tmp_vidlist

    return video_links


def download_from_links(links_dict, num=10, max_duration=1000):
    """

    :param num: int number of videos to download
    :param max_duration: int maximum seconds of video length
    :return: dir with downloaded videos

    TODO: Further processing to assure no duplicate videos are donwloaded
    """
    vid_dir = '../mp4videos/'
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)
    strms = []

    # Create streams
    for key in links_dict.keys():
        links = links_dict[key]
        if not isinstance(links, list):
            links = [links]
        for link in links:  # All links associated with keyword
            yt = pytube.YouTube(link)
            if int(yt.length) < max_duration:
                strm = yt.streams.first()
                strms.append([strm, key, yt.length])

    # shuffle list
    random.shuffle(strms)

    # download streams and rename
    if num is None:
        num = 1

    for vid in strms[:num]:
        stream = vid[0]
        filename = "bball_" + str(stream.itag) + '_dur_' + vid[2] + '.mp4'
        stream.download(output_path=vid_dir,
                        filename=filename)


def convert_to_frames(filename):
    vidcap = cv2.VideoCapture('../mp4videos/' + filename)
    success, image = vidcap.read()
    count = 0

    dir_to_save = '../images/' + filename.split('.')[0]

    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
        files = []
    else:
        files = os.listdir(dir_to_save)

    i = 0
    prenum = '00000'
    while success:
        length = len(str(i))
        count = prenum[0:5 - length] + str(i)
        img_filename = '/frame' + count + '.jpg'
        if img_filename not in files:
            cv2.imwrite(dir_to_save + img_filename, image)  # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
        i += 1

def clean_video():
    paths = []
    sub_dir = '../images/bball_22_dur_81mp4/'
    images = os.listdir('../images/bball_22_dur_81mp4/')

    for image_ in images:
        paths.append(sub_dir + image_)
    paths = sorted(paths)

    data_pix = {}
    data_pix[str(0)] = []
    data_pix[str(1)] = []
    data_pix[str(2)] = []


    for i, path in enumerate(paths, 0):

        img = image.load_img(path, target_size=(256, 256))
        img = np.array(img).transpose(2, 1, 0)
        for j in range(3):
            img_color = img[j, :, :]
            win_mean = np.mean(img_color)
            win_std = np.std(img_color)
            data_pix[str(j)].append([win_mean, win_std])

    i = 0
    outliers = []
    for r, g, b in zip(data_pix[str(0)], data_pix[str(1)], data_pix[str(2)]):
        if b[0] > g[0] and b[0] > r[0]:
            outliers.append(i)
        i += 1

    for index in outliers:
        img_name = paths[index]
        os.remove(img_name)

def save_data_csv(subdir):
    data_frame = []
    if not isinstance(subdir, list):
        subdir = [subdir]
    root_dir = '../images/'
    for video_frames in os.listdir(root_dir):
        # check if any selected prefixes are in directory str
        if any(video_frames in sub for sub in subdir):
            video_filename = os.path.join(root_dir, video_frames)
            img_names = os.listdir(video_filename)
            # append each path
            for images in img_names:
                data_frame.append([os.path.join(video_filename, images), video_filename])
    data_frame = sorted(data_frame, key=lambda x: x[0])
    pd_data_frame = pd.DataFrame(data_frame, columns=['video', 'class'])

    return pd_data_frame

if __name__ == '__main__':
    # links = get_youtube_links()
    #download_from_links({'url': 'https://www.youtube.com/watch?v=8skJ8wj9G_U'}, num=None)
    #filename = 'bball_22_dur_81mp4.mp4'
    #convert_to_frames(filename)
    #clean_video()
    df = save_data_csv('bball_22_dur_81mp4.mp4')
    with open('../pickle_data/vid1.pickle', 'wb') as handle:
        pickle.dump(df, handle)
    print(df.head())

