# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: cnnv_clustering.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-25
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

import time
import h5py

import numpy as np

from sklearn.cluster import KMeans

from utils.logger import logger
from utils.args import opt


def load_features(filepath):
    fp = h5py.File(filepath, mode='r')
    features = np.array(fp['features'][()]).squeeze()
    mean_feature = np.array(fp['mean_features'][()]).reshape(1, -1)
    fp.close()
    return features - mean_feature


def cnnv_clustering():
    filepath = os.path.join(opt['featurepath'], 'cnnlv-sampling-features.h5')
    features = load_features(filepath)

    kmeans = KMeans(n_clusters=opt['num_centers'],
                    random_state=0,
                    max_iter=10,
                    verbose=False,
                    n_init=3)
    start_t = time.time()
    kmeans.fit(features)
    end_t = time.time() - start_t
    logger.info('clustering done. time: {:.4f}(m)'.format(end_t / 60))

    centers = kmeans.cluster_centers_
    return centers


def main():
    centers = cnnv_clustering()
    filepath = os.path.join(opt['featurepath'], 'cnnv-centers.h5')
    fp = h5py.File(filepath, mode='w')
    fp.create_dataset(name='centers', data=centers)
    fp.close()
    logger.info('all done')


if __name__ == "__main__":
    main()


'''bash
python videoprocess/cnnv_clustering.py --dataname svd --approach cnnvcluster
'''
