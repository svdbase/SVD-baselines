# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: calc_hamming_ranking.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-10-16
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys

import numpy as np
import multiprocessing as mp

from sklearn.metrics import average_precision_score

from .args import opt
from .logger import logger


def __hamming_dist__(b1, b2):
    if b1.ndim == 1:
        b1 = b1.reshape(1, -1)
    if b2.ndim == 1:
        b2 = b2.reshape(1, -1)

    bit = b1.shape[1]
    hamm = (bit - b1.dot(b2.T)) / 2
    return hamm.squeeze()


class HammingRanking(object):
    def __init__(self, codes, unlabeled_keys, verbose=None):
        super(HammingRanking, self).__init__()
        self.verbose = verbose
        self.unlabeled_keys = unlabeled_keys
        self.codes = codes
        self.num_procs = opt['num_procs']
        aps = mp.Manager()
        self.aps = aps.list()
        self.verbose = verbose
        self.procs = []
        self.input = mp.Queue()

        for idx in range(self.num_procs):
            p = mp.Process(target=self.work, args=(idx, ))
            p.start()
            self.procs.append(p)

        logger.info('init done.')

    def process(self, params):
        ind, video, groundtruth = params
        y_true = []
        y_score = []

        for idx, cid in enumerate(groundtruth):
            y_true.append(groundtruth[cid])
            y_score.append(-__hamming_dist__(self.codes[video], self.codes[cid]))

        for idx, uid in enumerate(self.unlabeled_keys):
            if uid in groundtruth:
                continue

            y_true.append(0)
            y_score.append(-__hamming_dist__(self.codes[video], self.codes[uid]))

        ap = average_precision_score(y_true, y_score)
        if self.verbose:
            logger.info('idx: {:5d}, ap: {:.4f}, video: {}'.format(ind, ap, video))
        return ap

    def work(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                results = self.process(params)
                self.handle_result(results)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.info('Exception Type: {}, Filename: {}, Line: {}'.format(exc_type, fname, exc_tb.tb_lineno))
                raise print(e)

    def start(self, params):
        if isinstance(params, list):
            for idx, param in enumerate(params):
                param_ = idx, param
                self.input.put(param_)
        if isinstance(params, dict):
            for idx, key in enumerate(params):
                param_ = idx, key, params[key]
                self.input.put(param_)
        self.input.put(None)

    def stop(self):
        for idx, proc in enumerate(self.procs):
            proc.join()
            if self.verbose:
                logger.info('process: {} done'.format(idx))

    def handle_result(self, result):
        self.aps.append(result)

    def get_results(self):
        aps = np.array(list(self.aps))
        return np.mean(aps)


def calc_hamming_ranking(codes, unlabeled_keys, gnds, verbose=None):
    hr = HammingRanking(codes, unlabeled_keys, verbose)
    hr.start(gnds)
    hr.stop()
    map = hr.get_results()

    logger.info('MAP: {:.4f}'.format(map))
    return map


