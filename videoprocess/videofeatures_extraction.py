# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: videofeatures_extraction.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-25
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import h5py

parenddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parenddir)

import numpy as np
import multiprocessing as mp

from utils.util import *
from utils.args import opt
from utils.logger import logger


class VideoFeatureExtractor(object):
    def __init__(self):
        self.procs = []
        self.num_procs = opt['num_procs']

        vfeatures = mp.Manager()
        self.vfeatures = vfeatures.dict()

        self.input = mp.Queue()

        for idx in range(self.num_procs):
            p = mp.Process(target=self.worker, args=(idx, ))
            p.start()
            self.procs.append(p)

    @staticmethod
    def __normalize__(X):
        X -= X.mean(axis=1, keepdims=True)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-15
        return X

    def normalization(self, params):
        featurepath = os.path.join(opt['featurepath'], 'frames-features.h5')
        fp = h5py.File(featurepath, mode='r')
        index, video = params[0], params[1]
        framefeatures = np.array(fp[video][()]).squeeze()
        if framefeatures.ndim == 1:
            framefeatures = np.array([framefeatures, framefeatures])
        vfeature = self.__normalize__(framefeatures)
        vfeature = self.__normalize__(vfeature)
        vfeature = vfeature.mean(axis=0, keepdims=True)
        vfeature = self.__normalize__(vfeature)
        self.vfeatures[video] = vfeature
        if index % 10000 == 0:
            logger.info('index: {:6d}, video: {}'.format(index, video))

    def worker(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                self.normalization(params)
            except Exception as e:
                logger.info('Exception: {}. video: {}'.format(e, params[1]))

    def start(self, videolists):
        for idx, video in enumerate(videolists):
            self.input.put([idx, video])
        self.input.put(None)

    def stop(self):
        for idx, proc in enumerate(self.procs):
            proc.join()
            logger.info('process: {} is done.'.format(idx))

    def save_features(self):
        vfeatures = dict(self.vfeatures)
        vfeaturepath = os.path.join(opt['featurepath'], 'videos-features.h5')
        fp = h5py.File(vfeaturepath, mode='w')
        for video in vfeatures:
            feature = vfeatures[video]
            fp.create_dataset(name=video, data=feature)
        fp.close()
        logger.info('saving feature done')


def main():
    vfe = VideoFeatureExtractor()
    videos = get_video_id()
    logger.info('#video: {}'.format(len(videos)))
    vfe.start(videos)
    vfe.stop()
    vfe.save_features()
    logger.info('all done')


if __name__ == "__main__":
    main()

'''bash
python videoprocess/videofeatures_extraction.py --dataname svd-example
'''
