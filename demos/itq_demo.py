# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: itq_demo.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-9-15
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import random

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from approach.itq import ITQAlgo

from utils.util import *
from utils.args import opt
from utils.logger import logger
from utils.calc_hamming_ranking import calc_hamming_ranking

__DIM = 4096


def main():
    # basic setting
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])

    # create approach
    itq = ITQAlgo(__DIM)
    logger.info('creating itq method done')

    # load features
    featurepath = os.path.join(opt['featurepath'], 'videos-features.h5')
    features, mean_feature = load_features(featurepath)

    # train itq
    train_features = load_features(featurepath, dtype='array')
    itq.learn_hash_function(train_features)

    # generate binary codes
    codes = itq.generate_codes(features, mean_feature=mean_feature)
    logger.info('generating codes done')

    # load groundtruth and unlabeled-keys
    gnds = load_groundtruth('test_groundtruth')
    unlabeled_keys = get_video_id('unlabeled-data')
    logger.info('loading gnds and unlabeled keys done. #query: {}'.format(len(gnds)))

    # calculate map
    map = calc_hamming_ranking(codes, unlabeled_keys, gnds)
    logger.info('map: {:.4f}'.format(map))

    logger.info('all done')

if __name__ == "__main__":
    assert opt['approach'] == 'itq'
    main()


'''bash
python demos/itq_demo.py --dataname svd --approach itq
'''
