# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: util.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-26
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os

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
        for dtype in ['query', 'labeled-data', 'unlabeled-data']:
            filepath = os.path.join(opt['metadatapath'], dtype + '-id')
            with open(filepath, 'r') as fp:
                for tmps in fp:
                    videos.add(tmps.strip())
        return videos
