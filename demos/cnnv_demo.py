# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: cnnv_demo.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-9-15
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from utils.util import get_video_id, load_groundtruth, load_features
from utils.args import opt
from utils.logger import logger
from utils.calc_euclidean_search import calc_euclidean_search


def main():
    unlabeled_keys = get_video_id(dtype='unlabeled-data')

    gt_file = os.path.join(opt['metadatapath'], 'test_groundtruth')
    gnds = load_groundtruth(gt_file)

    featurepath = os.path.join(opt['featurepath'], 'cnnv-agg-features.h5')
    all_features, _ = load_features(featurepath)
    logger.info('load features done')

    map = calc_euclidean_search(all_features, unlabeled_keys, gnds)
    logger.info('map: {:.4f}'.format(map))
    logger.info('all done')


if __name__ == "__main__":
    main()


'''bash
python demos/cnnv_demo.py --dataname svd --approach cnnv
'''