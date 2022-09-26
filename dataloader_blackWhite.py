import gzip
import math
import numpy as np
import os
import cv2
import torch
import torch.utils.data as data
import pandas as pd
from os.path import join, exists

def createLabelsArray(unshaped_arr, seq_len):
    shaped_arr = np.reshape(unshaped_arr, (int(unshaped_arr.shape[0] / seq_len), seq_len, 3))
    # shape 2 is for fake or not fake, one hot vector
    labels_arr = np.zeros((shaped_arr.shape[0], 3))
    index = 0
    for seq in shaped_arr:
        label = unshaped_arr[index*seq_len][0]
        labels_arr[index] = unshaped_arr[index*seq_len]
        index += 1
    return labels_arr

def read_clip_images(path, image_size=128, vid_seq = 30):

    video_frames = []
    for image_name in sorted(os.listdir(path)):
        img = cv2.imread(join(path, image_name))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.moveaxis(np.array([gray_image]), 0, -1).repeat(3, 2)
        img[img > 240] = 255
        img[img <= 240] = 0

        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        video_frames.append(img)

    video_frames = np.array(video_frames)
    return video_frames[:vid_seq]


class VideoFramesDataloader(data.Dataset):
    def __init__(self, root, dataset_path, is_train=True, n_frames_input=15, n_frames_output=15, sequence_len=30,
                 transform=None, image_size=128):
        super(VideoFramesDataloader, self).__init__()

        self.dataset_path = dataset_path

        # ****************** MAKE IT DYNAMIC ******************8

        self.is_train = is_train
        self.video_dirs = os.listdir(dataset_path)
        self.length = len(self.video_dirs)
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        self.image_size = image_size

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        video_name = self.video_dirs[idx]
        video_folder_path = join(self.dataset_path, video_name)
        images = read_clip_images(video_folder_path, image_size=self.image_size, vid_seq=30)
        #print(sorted(os.listdir(video_folder_path))[0])
        images = images.transpose(0, 3, 1, 2)

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()

        out = [idx, input, output]
        return out

    def __len__(self):
        return self.length
