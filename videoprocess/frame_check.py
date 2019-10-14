# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: frame_check.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-9-28
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import h5py

import numpy as np
import multiprocessing as mp

from utils.util import *
from utils.args import opt
from utils.logger import logger


class FrameChecker:
    def __init__(self):
        super(FrameChecker, self).__init__()

        failed_log = mp.Manager()
        self.failed_log = failed_log.list()

        self.procs = []
        self.num_procs = opt['num_procs']
        self.input = mp.Queue()

        for idx in range(self.num_procs):
            p = mp.Process(target=self.worker, args=(idx, ))
            p.start()
            self.procs.append(p)

    def frame_checker(self, params):
        index, video = params[0], params[1]
        framepath = os.path.join(opt['framepath'], video)
        if not os.path.exists(framepath):
            self.failed_log.append(video)
            return
        infopath = os.path.join(framepath, 'info.h5')
        fo = h5py.File(infopath, mode='r')
        num_frames = fo['num_frames'][()]
        fo.close()
        if num_frames == 0:
            self.failed_log.append(video)
            return
        jpgflag = True
        for idx in range(num_frames):
            jpgfile = os.path.join(framepath, '{:04d}.jpg'.format(index))
            if not os.path.exists(jpgfile):
                jpgflag = False
                break
        if not jpgflag:
            self.failed_log.append(video)
            return

    def worker(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                self.frame_checker(params)
            except Exception as e:
                logger.info('Exception: {}.'.format(e))

    def start(self, videos):
        for idx, video in enumerate(videos):
            self.input.put([idx, video])
        self.input.put(None)

    def stop(self):
        for idx, proc in enumerate(self.procs):
            proc.join()
            logger.info('process: {} done'.format(idx))

    def get_results(self):
        failed_log = list(self.failed_log)
        if len(failed_log) > 0:
            filepath = os.path.join(opt['infopath'], 'failed-log')
            with open(filepath, 'w') as fp:
                for idx, failed in enumerate(failed_log):
                    logger.info('video: {} failed.'.format(failed))
                    fp.write(failed + '\n')


def main():
    videos = list(get_video_id())
    logger.info('#{} videos need to be processed'.format(len(videos)))

    fc = FrameChecker()
    fc.start(videos)
    fc.stop()
    fc.get_results()
    logger.info('all done')


if __name__ == "__main__":
    main()

'''bash
python videoprocess/frame_check.py --dataname svd-example
'''


