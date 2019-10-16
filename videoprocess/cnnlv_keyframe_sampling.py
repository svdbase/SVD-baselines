# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: cnnlv_keyframe_sampling.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-25
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import sys
import os
import h5py

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

import numpy as np
import multiprocessing as mp

from utils.util import get_video_id
from utils.args import opt
from utils.logger import logger


class CNNLVFrameSampler(object):
    def __init__(self, pos_labeled_keys=None):
        self.procs = []
        self.num_procs = opt['num_procs']
        self.pos_labeled_keys = pos_labeled_keys

        features = mp.Manager()
        self.features = features.list()

        self.input = mp.Queue()

        for idx in range(self.num_procs):
            p = mp.Process(target=self.work, args=(idx, ))
            p.start()
            self.procs.append(p)

    @staticmethod
    def __normalize__(X):
        X -= X.mean(axis=1, keepdims=True)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-15
        return X

    def reading(self, params):
        featurepath = os.path.join(opt['featurepath'], 'frames-features.h5')
        fp = h5py.File(featurepath, mode='r')
        idx, video = params[0], params[1]
        feature = np.array(fp[video][()]).squeeze().reshape(-1, 4096)
        self.features.append(feature)
        fp.close()
        if idx % 10000 == 0:
            logger.info('idx: {:6d}, video: {}'.format(idx, video))

    def work(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                self.reading(params)
            except Exception as e:
                logger.info('Exception: {}, wrong video: {}'.format(idx, params[1]))

    def start(self, video_lists):
        for idx, video in enumerate(video_lists):
            self.input.put([idx, video])
        self.input.put(None)

    def stop(self):
        for proc in self.procs:
            proc.join()

    def get_results(self):
        features = list(self.features)
        features = np.concatenate(features)
        num_frames = features.shape[0]
        ind = np.random.permutation(num_frames)
        features = features[ind[0: opt['num_key_frames']]]
        mean_feature = np.mean(features, axis=0)
        return features, mean_feature


def main():
    video_lists = get_video_id(dtype='labeled-data')
    pos_video_lists = None
    cfs = CNNLVFrameSampler(pos_video_lists)
    cfs.start(video_lists)
    cfs.stop()
    features, mean_feature = cfs.get_results()
    filepath = os.path.join(opt['featurepath'], 'cnnlv-sampling-features.h5')
    fp = h5py.File(filepath, mode='w')
    fp.create_dataset(name='features', data=features)
    fp.create_dataset(name='mean_features', data=mean_feature)
    fp.close()
    logger.info('all done')


if __name__ == "__main__":
    main()

'''bash
python videoprocess/cnnlv_keyframe_sampling.py --dataname svd --approach cfs
'''
