# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: itq.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-26
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np

from utils.args import opt


class ITQAlgo:
    bit = opt['bit']

    def __init__(self, dim):
        self.dim = dim
        self.w = None

    def learn_hash_function(self, train_features):
        c = np.cov(train_features.transpose())

        l, pc = np.linalg.eig(c)

        l_pc_ordered = sorted(zip(l, pc.transpose()), key=lambda _p: _p[0], reverse=True)
        pc_top = np.array([p[1] for p in l_pc_ordered[:self.bit]]).transpose()

        v = np.dot(train_features, pc_top)

        b, rotation = self.itq_rotation(v, 50)

        proj = np.dot(pc_top, rotation)
        self.w = proj

    def itq_rotation(self, v, n_iter):
        bit = self.bit
        r = np.random.randn(bit, bit)
        u11, s2, v2 = np.linalg.svd(r)

        r = u11[:, :bit]

        for i in range(n_iter):
            z = np.dot(v, r)
            ux = np.ones(z.shape) * (-1.)
            ux[z >= 0] = 1
            c = np.dot(ux.transpose(), v)
            ub, sigma, ua = np.linalg.svd(c)
            r = np.dot(ua, ub.transpose())
        z = np.dot(v, r)
        b = np.ones(z.shape) * -1.
        b[z >= 0] = 1
        return b, r

    def generate_codes(self, features, mean_feature):
        assert self.w is not None

        if isinstance(features, np.ndarray):
            if mean_feature is not None:
                features = features - mean_feature
            codes = np.sign(np.dot(features, self.w))
            return codes
        if isinstance(features, dict):
            codes = {}
            if mean_feature is None:
                mean_feature = 0.
            for video in features:
                feature = features[video] - mean_feature
                codes[video] = np.sign(np.dot(feature, self.w))
            return codes
