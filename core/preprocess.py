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

    query_link = "https://www.youtube.com/results?search_query=NBA+basketball+highlights+"
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
            if max_duration is None:
                strm = yt.streams.first()
                strms.append([strm, key, yt.length])
            elif int(yt.length) < max_duration:
                strm = yt.streams.first()
                strms.append([strm, key, yt.length])

    # shuffle list

    # download streams and rename
    if num is None:
        num = 1

    if max_duration is not None:
        random.shuffle(strms)

    for vid in strms[:num]:
        stream = vid[0]
        save_filename = "bball_" + str(stream.itag) + '_dur_' + vid[2]
        filename = save_filename + '.mp4'
        stream.download(output_path=vid_dir,
                        filename=save_filename)
    return filename


def convert_to_frames(filename):
    # Create directory to save video
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
            #print('Read a new frame: ', success)
        i += 1

def clean_video(filename):
    paths = []
    # directory where images were stored
    sub_dir = '../images/' + filename.split('.')[0] +'/'
    images = os.listdir(sub_dir)

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

def save_data_csv(subdir, filename):
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
        else:
            print('some files are not in /images/ directory')

    data_frame = sorted(data_frame, key=lambda x: x[0])
    pd_data_frame = pd.DataFrame(data_frame, columns=['video', 'class'])

    with open('../pickle_data/' + filename, 'wb') as handle:
        pickle.dump(pd_data_frame, handle)

    return pd_data_frame

def split_test_train_data(pickle_filename, fps=30, scale=0.20):
    """ Splits data into training and testing
    :param pickle_filename: str pickle filename where paths are store
    :param fps: int frames per second, default is 30 (decides the a

    :returns None
    """

    path_to_pickle_file = '../pickle_data/' + pickle_filename
    assert os.path.exists(path_to_pickle_file), "Pickle file does not exist"
    assert 0 < fps <= 32, "fps needs to be between 0 and 31 "

    # load data
    with open(path_to_pickle_file, 'rb') as handle:
        data_frame = pickle.load(handle)
    print(len(data_frame))

    filtered_data = []
    interval_size = int(32 // fps)
    for i in range(len(data_frame)):
        val = i % 32
        check = val % interval_size
        if check == 0:
            val_item = data_frame.iloc[i]
            filtered_data.append([val_item['video'], val_item['class']])

    # split data (20 percent training)
    N = len(filtered_data)
    number_of_videos = N // 32
    number_test_videos = int(scale * number_of_videos)

    interval_split = number_of_videos // number_test_videos

    print("Training Videos: " + str(number_of_videos - number_test_videos) + ', Test Videos: ' + str(number_test_videos))

    testlist = []
    trainlist = []

    for i in range(number_of_videos):
        if i % interval_split == 0:
            testlist.extend(filtered_data[i*32: i*32 + 32])
        else:
            trainlist.extend(filtered_data[i*32: i*32 + 32])


    test_df = pd.DataFrame(testlist, columns=['video', 'class'])
    train_df = pd.DataFrame(trainlist, columns=['video', 'class'])

    dir_test, dir_train = '../pickle_data/testing/test_' + str(fps) + 'fps_',\
                          '../pickle_data/training/train_' + str(fps) + 'fps_'

    with open(dir_test + pickle_filename, 'wb') as tst, open(dir_train + pickle_filename, 'wb') as trn:
        pickle.dump(test_df, tst)
        pickle.dump(train_df, trn)

    return test_df, train_df


if __name__ == '__main__':
    #links = get_youtube_links(keywords=['2017'])
    #filename = download_from_links({'slow_mode':'https://www.youtube.com/watch?v=0ePGDpSBlmY'}, num=None, max_duration=None)
    #filename = download_from_links({'slow_mode': 'https://www.youtube.com/watch?v=0ePGDpSBlmY'}, num=1,
    #                             max_duration=None)
    #convert_to_frames('nba_slow_mode.mp4')
    #clean_video(filename)
    df = save_data_csv(['nba_slow_mode'], filename='nba_slow_mode.pickle')

    fps_ = [16]

    for fps in fps_:
        tt, t = split_test_train_data('nba_slow_mode.pickle', fps=fps, scale=0.10)

