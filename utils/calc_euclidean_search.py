# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: calc_euclidean_search.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-9-15
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os

import numpy as np
import multiprocessing as mp

from scipy.spatial.distance import euclidean
from sklearn.metrics import average_precision_score

from utils.args import opt
from utils.logger import logger


class EuclideanSearch(object):
    def __init__(self, features, unlabel_keys, verbose=None):
        super(EuclideanSearch, self).__init__()
        self.features = features
        self.unlabeled_keys = unlabel_keys

        aps = mp.Manager()
        self.aps = aps.list()

        self.input = mp.Queue()
        self.num_procs = opt['num_procs']
        self.procs = []
        self.verbose = verbose

        for idx in range(self.num_procs):
            p = mp.Process(target=self.worker, args=(idx, ))
            p.start()
            self.procs.append(p)

    def worker(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                ap = self.calc_ap(params)
                self.aps.append(ap)
            except Exception as e:
                logger.info('Exception: {}.'.format(e))

    def calc_ap(self, params):
        index, video, gnd = params[0], params[1], params[2]
        query_feature = self.features[video]

        y_true, y_score = [], []
        for idx, cid in enumerate(gnd):
            g = gnd[cid]
            s = euclidean(self.features[cid], query_feature)
            y_true.append(g)
            y_score.append(-s)

        for idx, uid in enumerate(self.unlabeled_keys):
            s = euclidean(self.features[uid], query_feature)
            y_score.append(-s)
            y_true.append(0)

        ap = average_precision_score(y_true, y_score)
        if self.verbose:
            logger.info('ap: {:.4f}@idx: {}, video: {}.'.format(ap, index, video))
        return ap

    def start(self, gnds):
        for idx, video in enumerate(gnds):
            gnd = gnds[video]
            self.input.put([idx, video, gnd])
        self.input.put(None)

    def stop(self):
        for idx, proc in enumerate(self.procs):
            proc.join()
            if self.verbose:
                logger.info("process: {} done".format(idx))

    def get_results(self):
        aps = list(self.aps)
        map = np.mean(np.array(aps))
        return map


def calc_euclidean_search(features, unlabeled_keys, gnds, verbose=None):
    es = EuclideanSearch(features, unlabeled_keys, verbose=verbose)
    es.start(gnds)
    es.stop()
    map = es.get_results()
    return map


