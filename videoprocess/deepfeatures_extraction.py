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

import h5py
import torch

import numpy as np

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from utils.util import *
from utils.args import opt
from utils.logger import logger
from modules.vgg import vgg16
from data.create_loader import create_loader

__filelists = [
    'frames-0.h5',
    'frames-1.h5',
    'frames-2.h5',
    'frames-3.h5',
    'frames-4.h5',
    'frames-5.h5',
    'frames-6.h5',
    'frames-7.h5',
    'frames-8.h5',
    'frames-9.h5',
]


def create_imagepaths():
    imagespaths = []
    cutoffs = []
    for filelist in __filelists:
        filepath = os.path.join(opt['framepath'], filelist)
        fp = h5py.File(filepath, mode='r')
        for video in fp:
            fg = fp[video]
            num_frame = int(np.array(fg['num_frames'][()]))
            imagepaths = []
            for frame in fg:
                if frame != 'num_frames':
                    imagepaths.append([filelist, video, frame])
            assert len(imagepaths) == num_frame
            imagespaths += imagepaths
            cutoffs.append([video, num_frame])
        fp.close()
    return imagespaths, cutoffs


def main():
    batch_size = 60
    imagespaths, cutoffs = create_imagepaths()
    logger.info('#{} frames will be processed.'.format(len(imagespaths)))
    logger.info('#{} videos will be processed.'.format(len(cutoffs)))

    dataloader = create_loader(imagespaths, batch_size=batch_size)

    model = vgg16(pretrained=True).to(opt['device'])
    model.eval()

    featurepath = os.path.join(opt['featurepath'], 'frames-features.h5')
    fp = h5py.File(featurepath, mode='w')

    count = 0
    buffer = []
    with torch.no_grad():
        for iter, batch in enumerate(dataloader, 0):
            frames, index = batch
            frames, index = frames.to(opt['device']), index.numpy()
            features = list(model(frames).data.cpu().numpy())
            current_len = len(features) + len(buffer)
            buffer += features
            if current_len >= cutoffs[count][1]:
                while True:
                    if count < len(cutoffs) and len(buffer) >= cutoffs[count][1]:
                        offset = cutoffs[count][1]
                        save_features = buffer[:offset]
                        if not len(save_features) == cutoffs[count][1]:
                            logger.info('Length Error: {}/{}'.format(len(save_features), cutoffs[count][1]))
                            return
                        video = cutoffs[count][0]
                        fp.create_dataset(name=video, data=np.array(save_features).astype(np.float32))

                        buffer = buffer[offset:]
                        count += 1
                        if opt['verbose']:
                            logger.info('video: {}, {:6d}/{:6d} is done.'.format(video, count, len(cutoffs)))
                    else:
                        break
    fp.close()
    logger.info('all done')


if __name__ == "__main__":
    main()


'''bash
CUDA_VISIBLE_DEVICES=1 python videoprocess/deepfeatures_extraction.py.py --dataname svd-example
'''

