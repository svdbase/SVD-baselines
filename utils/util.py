# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: util.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-26
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import h5py

import numpy as np

from collections import OrderedDict

from utils.args import opt


def load_features(featurepath, dtype='dict'):
    if dtype == 'dict':
        features = {}
        fo = h5py.File(featurepath, mode='r')
        mean_feature = 0.
        cnt = 0.
        for k in fo:
            feature = fo[k][()].squeeze().reshape(1, -1)
            mean_feature += feature
            cnt += 1
            features[k] = feature
        fo.close()
        mean_feature /= cnt
        return features, mean_feature
    if dtype == 'array':
        features = []
        fo = h5py.File(featurepath, mode='r')
        mean_feature = 0.
        cnt = 0.
        for k in fo:
            feature = fo[k][()].squeeze().reshape(1, -1)
            mean_feature += feature
            cnt += 1
            features.append(feature)
        fo.close()
        mean_feature /= cnt
        features = np.array(features).squeeze() - mean_feature
        return features


def get_video_id(dtype=None):
    if dtype is not None:
        videos = set()
        filepath = os.path.join(opt['metadatapath'], dtype + '-id')
        with open(filepath, 'r') as fp:
            for tmps in fp:
                videos.add(tmps.strip())
        return videos
    else:
        videos = set()
        filepath = os.path.join(opt['metadatapath'], 'all-video-id')
        with open(filepath, 'r') as fp:
            for tmps in fp:
                videos.add(tmps.strip())
        return videos


def load_groundtruth(filename=None):
    if filename is None:
        filename = 'groundtruth'
    filepath = os.path.join(opt['metadatapath'], filename)
    gnds = OrderedDict()
    with open(filepath, 'r') as fp:
        for idx, lines in enumerate(fp):
            tmps = lines.strip().split(' ')
            qid = tmps[0]
            cid = tmps[1]
            gt = int(tmps[-1])
            if qid not in gnds:
                gnds[qid] = {cid: gt}
            else:
                gnds[qid][cid] = gt
    return gnds