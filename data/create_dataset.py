# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: create_dataset.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-9-12
# +++++++++++++++++++++++++++++++++++++++++++++++++++
from .dataset import ImageDataset


def create_dataset(imagepaths):
    dataset = ImageDataset(imagepaths)
    return dataset
