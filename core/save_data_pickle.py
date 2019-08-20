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
    data_frame = sorted(data_frame, key=lambda x: x[0])
    pd_data_frame = pd.DataFrame(data_frame, columns=['video', 'class'])

    return pd_data_frame


root_dir = '/Users/jesusnavarro/Desktop/gan_video/UAV123/data_seq/UAV123/'
df = save_data_csv('bike', root_dir)


with open('/Users/jesusnavarro/Desktop/gan_video/pickle_data/test_boat_videos.pickle', 'wb') as handle:
    pickle.dump(df, handle)
print(df.head())