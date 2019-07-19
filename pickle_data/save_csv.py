import os
import pandas as pd
import pickle

def save_data_csv(prefix, root_dir):
    data_frame = []
    if not isinstance(prefix, list):
        prefix = [prefix]
    for sub_dir in os.listdir(root_dir):
        # check if any selected prefixes are in directory str
        if any(pre in sub_dir for pre in prefix):
            class_dir = os.path.join(root_dir, sub_dir)
            img_names = os.listdir(class_dir)
            # append each path
            for images in img_names:
                data_frame.append([os.path.join(class_dir, images), sub_dir])

    video_data_frame = []
    # group into frames of 32
    tmp_video = []
    for i, frame in enumerate(data_frame):
        if (i + 1) % 33 == 0:
            video_data_frame.append([tmp_video])
            tmp_video = []
        else:
            tmp_video.append(frame)

    # remove last video if len < 32

    if len(video_data_frame[-1]) < 32:
        video_data_frame = video_data_frame[:-2]


    pd_data_frame = pd.DataFrame(video_data_frame, columns=['video'])
    return pd_data_frame


root_dir = '/Users/jesusnavarro/Desktop/gan_video/UAV123/data_seq/UAV123/'
df = save_data_csv('bike', root_dir)

print(df.head())

with open('/Users/jesusnavarro/Desktop/gan_video/pickle_data/test_2000_videos.pickle', 'wb') as handle:
    pickle.dump(df[:2000], handle)
print(df.head())