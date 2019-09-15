# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: util.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-26
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from collections import OrderedDict

from utils.args import opt


def get_video_id(dtype=None):
    if dtype is not None:
        videos = set()
        filepath = os.path.join(opt['metadatapath'] + '-id')
        with open(filepath, 'r') as fp:
            for tmps in fp:
                videos.add(tmps.strip())

    else:
        videos = set()
        filepath = os.path.join(opt['metadatapath'], 'all-video-id')
        with open(filepath, 'r') as fp:
            for tmps in fp:
                videos.add(tmps.strip())
        return videos


def load_groundtruth(filename=None):
    if filename is None:
        filename = 'test_groundtruth'
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