# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: video_durations.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-26
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import pickle

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

import cv2
import multiprocessing as mp
import numpy as np

from utils.util import *
from utils.args import opt
from utils.logger import logger


class VideoDuration(object):
    def __init__(self, verobose=None):
        super(VideoDuration, self).__init__()
        self.verbose = verobose

        self.procs = []
        self.num_procs = opt['num_procs']

        durations = mp.Manager()
        self.durations = durations.dict()

        failed_videos = mp.Manager()
        self.failed_videos = failed_videos.list()

        self.input = mp.Queue()

        for idx in range(self.num_procs):
            p = mp.Process(target=self.worker, args=(idx, ))
            p.start()
            self.procs.append(p)

    def video_duration(self, params):
        index, video = params[0], params[1]
        videopath = os.path.join(opt['videopath'], video)

        try:
            v = cv2.VideoCapture(videopath)
            cnt = v.get(cv2.CAP_PROP_FPS)
            num_frames = v.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = num_frames / cnt
            if self.verbose:
                logger.info('index: {:6d}, video: {}, duration: {:.3f}(s)'.
                            format(index, video, duration))

            return video, duration
        except Exception as e:
            logger.info(e)

    def worker(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                video, duration = self.video_duration(params)
                if isinstance(duration, float) and duration > 0:
                    self.durations[video] = duration
                else:
                    self.failed_videos.append(video)

            except Exception as e:
                self.failed_videos.append(video)
                logger.info('Exception: {}, wrong video: {}'.
                            format(e, video))

    def start(self, videos):
        for idx, video in enumerate(videos):
            self.input.put([idx, video])
        self.input.put(None)

    def stop(self):
        for idx, proc in enumerate(self.procs):
            proc.join()
            logger.info('process: {} done.'.format(idx))

    def get_results(self):
        durations = dict(self.durations)
        failed_videos = list(self.failed_videos)
        num_processed = len(durations)
        tot_duration = np.sum(np.array(list(durations.values())))
        avg_duration = tot_duration / num_processed
        logger.info('#{:6d} videos are processed. Total durations: {:.4f}, Average durations: {:.4f}'.
                    format(num_processed, tot_duration, avg_duration))

        if len(failed_videos) > 0:
            logger.info('#{:6d} videos are failed.'.format(len(failed_videos)))
            with open(os.path.join(opt['infopath'], 'duration-failed.log'), 'w') as fp:
                for video in failed_videos:
                    fp.write(video + '\n')

        with open(os.path.join(opt['infopath'], 'durations.pkl'), 'wb') as fp:
            pickle.dump(durations, fp)


def main():
    videos = get_video_id()
    duration = VideoDuration(opt['verbose'])
    duration.start(videos)
    duration.stop()
    duration.get_results()
    logger.info('all done')


if __name__ == "__main__":
    main()


'''bash
python videoprocess/video_durations.py --dataname svd
'''



