# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: dataset.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-9-12
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import cv2
import h5py

import numpy as np
import torch.utils.data as data

from utils.args import opt


class ImageDataset(data.Dataset):
    __desired_size_ = (224, 224)

    def __init__(self,
                 imagepaths,
                 ):
        self.imagepaths = imagepaths

    @staticmethod
    def __decodejpeg(jpeg):
        x = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    def __getitem__(self, index):
        info = self.imagepaths[index]
        filename, video, key = info[0], info[1], info[2]
        fp = h5py.File(os.path.join(opt['framepath'], filename), mode='r')
        fg = fp[video]
        jpeg = np.array(fg[key][()])
        image = cv2.resize(self.__decodejpeg(jpeg), dsize=self.__desired_size_).transpose(2, 0, 1)
        fp.close()
        return image.astype(np.float32), index

    def __len__(self):
        return len(self.imagepaths)



