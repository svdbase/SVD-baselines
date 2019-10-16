# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: cnnv_feature_aggregation.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-25
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import h5py

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

import numpy as np
import multiprocessing as mp

from scipy.spatial.distance import cdist

from utils.util import get_video_id
from utils.args import opt
from utils.logger import logger


def nearest_neighbor(feature, centers):
    feature = feature.reshape(-1, 4096)
    if feature.shape[0] == 1:
        inner = feature.dot(centers.T).squeeze()
        norm_c = np.sum(centers ** 2, axis=1)
        dist = -2 * inner + norm_c
        idx = np.argmin(dist)
    else:
        dist = cdist(feature, centers)
        idx = np.argmin(dist, axis=1)
    return centers[idx]


def __load_cnnv_centers__():
    centerpath = os.path.join(opt['featurepath'], 'cnnv-centers.h5')
    fp = h5py.File(centerpath, mode='r')
    centers = np.array(fp['centers'][()]).squeeze()
    fp.close()
    return centers


class CNNVFeatureAggregation(object):
    def __init__(self, centers):
        self.centers = centers
        self.num_procs = opt['num_procs']

        self.input = mp.Queue()
        features = mp.Manager()
        self.features = features.dict()
        self.procs = []

        for idx in range(self.num_procs):
            p = mp.Process(target=self.work, args=(idx, ))
            p.start()
            self.procs.append(p)

    def feature_aggregations(self, params):
        featurepath = os.path.join(opt['featurepath'], 'frames-features.h5')
        fp = h5py.File(featurepath, mode='r')
        idx, video = params
        features = np.array(fp[video][()]).squeeze().reshape(-1, 4096)
        if features.ndim == 1:
            features = np.array([features, features])
        fp.close()

        sele_centers = nearest_neighbor(features, self.centers)
        agg_features = np.mean(sele_centers, axis=0)
        if idx % 10000 == 0:
            logger.info('idx: {:5d}, video: {}'.format(idx, video))
        return video, agg_features

    def work(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                video, features = self.feature_aggregations(params)
                self.features[video] = features
            except Exception as e:
                logger.info('Exception: {}'.format(e))

    def start(self, video_lists):
        for idx, video in enumerate(video_lists):
            params = idx, video
            self.input.put(params)
        self.input.put(None)

    def stop(self):
        for proc in self.procs:
            proc.join()

    def get_results(self):
        features = dict(self.features)
        filepath = os.path.join(opt['featurepath'], 'cnnv-agg-features.h5')

        fp = h5py.File(filepath, mode='w')
        for video in features:
            fp.create_dataset(name=video, data=features[video])
        fp.close()
        logger.info('cnnv aggregated features are save to:{}'.format(filepath))


def main():
    centers = __load_cnnv_centers__()
    cnnvfa = CNNVFeatureAggregation(centers)
    video_lists = get_video_id()
    cnnvfa.start(video_lists)
    cnnvfa.stop()
    cnnvfa.get_results()
    logger.info('all done')


if __name__ == "__main__":
    main()

'''bash
python videoprocess/cnnv_feature_aggregation.py --dataname svd --approach cnnvfa
'''
