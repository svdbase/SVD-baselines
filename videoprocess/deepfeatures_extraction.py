# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: deepfeatures_extraction.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-25
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import sys
import os
import cv2

import h5py
import torch

import numpy as np

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from utils.util import *
from utils.args import opt
from utils.logger import logger
from data.dataset import create_loader
from modules.vgg import vgg16


def create_image_list():
    def __load_image_paths__(framepath):
        infofile = os.path.join(framepath, 'info.h5')
        fp = h5py.File(infofile, mode='r')
        num_frames = int(fp['num_frames'][()])
        fp.close()
        images_paths = []
        for ii in range(num_frames):
            image_path = os.path.join(framepath, '{:04d}.jpg'.format(ii))
            images_paths.append(image_path)
        return images_paths
    videos = get_video_id()
    logger.info('#videos: {}'.format(len(videos)))
    images_paths = []
    cutoffs = []
    num_processing = 0.
    for idx, video in enumerate(videos):
        frame_path = os.path.join(opt['framepath'], video)
        feature_path = os.path.join(opt['featurepath'], video + '.h5')
        if not os.path.exists(feature_path):
            image_paths = __load_image_paths__(frame_path)
            if len(image_paths) < 1:
                continue
            images_paths += image_paths
            cnt = len(image_paths)
            cutoffs.append([cnt, video])
            num_processing += 1
    logger.info('#processing videos: {}'.format(num_processing))
    return images_paths, cutoffs


def check_images(image_paths):
    for imagepath in image_paths:
        if not os.path.join(imagepath):
            logger.info('file: {} not found'.format(imagepath))
            return False
    return True


def __normlize__(X):
    X -= X.mean(axis=1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-15
    return X


def deep_feature_extraction():
    batch_size = 400
    images_paths, cutoffs = create_image_list()
    logger.info('load image paths done. #video: {:5d}'.format(len(cutoffs)))

    dataloader = create_loader(images_paths, batch_size)
    logger.info('create data loader done')

    model = vgg16(pretrained=True).to(opt['device'])
    model.eval()
    logger.info('create cnn-model done')

    count = 0
    buffer = []
    filepath = os.path.join(opt['featurepath'], 'frames-features.h5')
    fw = h5py.File(filepath, mode='w')
    with torch.no_grad():
        for iter, batch in enumerate(dataloader, 0):
            images, ind = batch
            images, ind = images.to(opt['device']), ind.numpy()
            features = list(model(images).data.cpu().numpy())
            current_len = len(features) + len(buffer)
            buffer += features
            if current_len >= cutoffs[count][0]:
                while True:
                    if count < len(cutoffs) and len(buffer) >= cutoffs[count][0]:
                        offset = cutoffs[count][0]

                        next_video = ''
                        try:
                            curr_ind = ind[0] - len(buffer) + len(features) + offset - 1
                            next_ind = curr_ind + 1

                            curr_video = images_paths[curr_ind].split('/')[-2]
                            next_video = images_paths[next_ind].split('/')[-2]
                            if curr_video == next_video:
                                logger.info('Split Error: {}/{}'.format(curr_video, next_video))
                                return
                        except:
                            pass

                        save_features = buffer[:offset]
                        if not len(save_features) == cutoffs[count][0]:
                            logger.info('Length Error: {}/{}'.format(len(save_features), cutoffs[count][0]))
                            return
                        save_features = np.array(save_features)
                        video = cutoffs[count][1]
                        fw.create_dataset(name=video, data=save_features)
                        if opt['verbose'] and count % opt['output_period'] == 0:
                            logger.info('CNT: {:6d}/{:6d}, Video: {}, Next Video: {}'.format(
                                    count, len(cutoffs), video, next_video
                                ))

                        buffer = buffer[offset:]
                        count += 1
                    else:
                        break
    fw.close()


def main():
    deep_feature_extraction()


if __name__ == '__main__':
    main()

'''bash
CUDA_VISIBLE_DEVICES=12 python videoprocess/deepfeatures_extraction.py --dataname svd-example
'''

