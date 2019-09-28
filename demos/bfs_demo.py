# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: bfs_demo.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-9-15
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import h5py

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from utils.util import load_groundtruth, get_video_id
from utils.args import opt
from utils.logger import logger
from utils.calc_euclidean_search import calc_euclidean_search


def load_features():
    filepath = os.path.join(opt['featurepath'], 'videos-features.h5')
    # filepath = '/data3/jiangqy/dataset/svd/vfeatures_4096/video-features.h5'
    features = {}
    fp = h5py.File(filepath, mode='r')
    for k in fp:
        features[k] = fp[k][()]
    fp.close()
    return features


def main():
    ### load features
    features = load_features()
    logger.info('loading features done. #videos: {}'.format(len(features)))

    ### load groundtruth and unlabeled-keys
    gnds = load_groundtruth('test_groundtruth')
    unlabeled_keys = get_video_id('unlabeled-data')
    logger.info('load gnds and unlabeled keys done. #query: {}'.format(len(gnds)))

    ### calculate map
    map = calc_euclidean_search(features, unlabeled_keys, gnds)
    logger.info('map: {:.4f}'.format(map))

    logger.info('all done')


if __name__ == "__main__":
    main()


'''bash
python demos/bfs_demo.py --dataname svd-example
'''

