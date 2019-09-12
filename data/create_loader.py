# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: create_loader.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-9-12
# +++++++++++++++++++++++++++++++++++++++++++++++++++
from torch.utils.data import DataLoader
from .create_dataset import create_dataset


def create_loader(imagepaths, batch_size, shuffle=False, num_workers=10):
    dataset = create_dataset(imagepaths)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers
                            )
    return dataloader

