# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: isoh.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-26
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy import linalg

from utils.args import opt


class IsoHAlgo(object):
    bit = opt['bit']

    def __init__(self, dim):
        super(IsoHAlgo, self).__init__()
        self.dim = dim
        self.w = None

    def learn_hash_function(self, features):
        XX = np.cov(features.T)

        eigvals, eigvectors = np.linalg.eig(XX)
        eigval_eigvec_sorted = sorted(zip(eigvals, eigvectors.transpose()), key=lambda _p: _p[0], reverse=True)
        eigvectors_top = np.array([p[1] for p in eigval_eigvec_sorted[:self.bit]]).transpose()

        eigvals_top = np.array([p[0] for p in eigval_eigvec_sorted[:self.bit]])
        m = np.diag(eigvals_top[:self.bit])
        q = self.__lift_projection__(m, 50)
        proj = eigvectors_top.dot(q)
        self.w = proj

    @staticmethod
    def __lift_projection__(M, iter):
        def __eigsdescend__(X, n):
            D, V = np.linalg.eig(X)
            D = D[:n]
            V = V[:n]
            ind = np.argsort(-D.squeeze())
            D = D[ind]
            V = V[:, ind]
            return V, D

        n = M.shape[0]
        a = np.trace(M) / n
        R = np.random.randn(n, n)
        U, Sigma, VT = linalg.svd(R)

        Z = U.dot(M).dot(U.T)

        for ii in range(iter):
            T = Z
            for j in range(n):
                T[j, j] = a
            Q, _ = __eigsdescend__(T, n)
            Z = Q.dot(M).dot(Q.T)
        return Q.transpose()

    def generate_codes(self, features, mean_feature):
        assert self.w is not None
        codes = {}
        for video in features:
            feature = features[video] - mean_feature
            codes[video] = np.sign(np.dot(feature, self.w))
        return codes
