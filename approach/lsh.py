# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: lsh.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-26
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np

from utils.args import opt


class LSHAlgo:
    bit = opt['bit']

    def __init__(self, dim):
        self.dim = dim
        self.w = np.random.randn(self.dim, self.bit)

    def generate_codes(self, features, mean_feature=None):
        if isinstance(features, np.ndarray):
            if mean_feature is not None:
                features = features - mean_feature
            codes = np.sign(np.dot(features, self.w))
            return codes
        if isinstance(features, dict):
            codes = {}
            for video in features:
                feature = features[video] - mean_feature
                codes[video] = np.sign(np.dot(feature, self.w))
            return codes
