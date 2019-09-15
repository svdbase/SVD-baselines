# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: frame_extraction.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-25
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import h5py
import cv2
import imageio
import skimage
import pickle

import numpy as np
import multiprocessing as mp

from utils.util import *
from utils.args import opt
from utils.logger import logger


class FrameExtractor:
    def __init__(self,
                 verbose=None
                 ):
        super(FrameExtractor, self).__init__()
        self.verbose = verbose

        self.procs = []
        self.fps = opt['fps']
        self.num_procs = opt['num_procs']

        self.processed = self.load_processed()
        logger.info('#processed: {}'.format(len(self.processed)))
        processing = mp.Manager()
        self.processing = processing.dict()

        self.input = mp.Queue()

        for idx in range(self.num_procs):
            p = mp.Process(target=self.worker, args=(idx, ))
            p.start()
            self.procs.append(p)

    @staticmethod
    def load_processed():
        filename = 'processed.pkl'
        filepath = os.path.join(opt['framepath'], filename)
        processed = {}
        if os.path.exists(filepath):
            with open(filepath, 'rb') as fp:
                processed = pickle.load(fp)
        return processed

    @staticmethod
    def cv2_extractor(videopath, fps=1):
        try:
            cap = cv2.VideoCapture(videopath)
            cnt = cap.get(cv2.CAP_PROP_FPS)
            count, num_frame = 0, 0
            jpegs = []
            while cap.isOpened():
                ret, frame = cap.read()
                if isinstance(frame, np.ndarray):
                    if int(count * fps / (round(cnt))) == 0:
                        jpeg = cv2.imencode('.jpg', frame)[1]
                        jpegs.append(jpeg)
                        num_frame += 1
                else:
                    break
                count += 1
            if num_frame > 0:
                return num_frame, jpegs
        except Exception as e:
            print('Exception: {} (cv2)'.format(e))
            return None

    @staticmethod
    def ffmpeg_extractor(videopath, fps):
        try:
            reader = imageio.get_reader(videopath, 'ffmpeg')
            duration = reader.get_meta_data()['duration']
            num_frame = 0
            jpegs = []
            for count, im in enumerate(reader):
                image = skimage.img_as_uint(im).astype(np.uint8)
                if int(count * fps / (round(duration))) == 0:
                    jpeg = cv2.imencode('.jpg', image)[1]
                    jpegs.append(jpeg)
                    num_frame += 1
            if num_frame > 0:
                return num_frame, jpegs
        except Exception as e:
            print('Exception: {} (ffmpeg)'.format(e))
            return None

    def frame_extractor(self, params):
        idx, video_piece = params[0], params[1]
        logger.info('process: {}, processing: {}'.format(idx, len(video_piece)))
        filename = '-'.join(['frames', str(idx) + '.h5'])
        filepath = os.path.join(opt['framepath'], filename)
        fp = h5py.File(filepath, mode=_write_mode)
        num_videos = len(video_piece)
        for ind, video in enumerate(video_piece):
            if video in self.processed:
                self.processing[video] = filepath
                continue
            videopath = os.path.join(opt['videopath'], video)
            results = self.cv2_extractor(videopath, self.fps)
            if results is not None:
                num_frame, jpegs = results
                fg = fp.create_group(name=video)
                fg.create_dataset(name='num_frames', data=num_frame)
                for index, jpeg in enumerate(jpegs):
                    fg.create_dataset(name=str(index), data=jpeg)
                self.processing[video] = filepath
                if self.verbose:
                    logger.info('filepart: {:2d}, file: {:5d}/{:5d}. frame extraction for video {} is done by cv2'.format(
                        idx, ind, num_videos, video))
            else:
                results = self.ffmpeg_extractor(videopath, self.fps)
                if results is not None:
                    num_frame, jpegs = results
                    fg = fp.create_group(name=video)
                    fg.create_dataset(name='num_frames', data=num_frame)
                    for index, jpeg in enumerate(jpegs):
                        fg.create_dataset(name=str(index), data=jpeg)
                    self.processing[video] = filepath
                    if self.verbose:
                        logger.info('filepart: {:2d}, file: {:5d}/{:5d}. '
                                    'frame extraction for video {} is done by ffmpeg'.format(
                            idx, ind, num_videos, video))
                else:
                    logger.info('processing video: {} failed'.format(video))
        fp.close()

    def worker(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                self.frame_extractor(params)
            except Exception as e:
                logger.info('Exception: {}.'.format(e))

    def start(self, video_pieces):
        for idx, video_piece in enumerate(video_pieces):
            self.input.put([idx, video_piece])
        self.input.put(None)

    def stop(self):
        for idx, proc in enumerate(self.procs):
            proc.join()
            logger.info('process: {} done'.format(idx))

        processing = dict(self.processing)
        logger.info('#{} videos are extracted successfully.'.format(len(processing)))
        filename = 'processed.pkl'
        filepath = os.path.join(opt['framepath'], filename)
        with open(filepath, 'wb') as fp:
            pickle.dump(processing, fp)

_NUM_FRAME_PIECE = 10
_write_mode = 'w' ## or 'a'


def split_videos(videos):
    def split_list(templist, n):
        m = int(len(templist) / n) + 1
        for i in range(0, len(templist), m):
            yield templist[i: i+m]
    videos_pieces = split_list(videos, _NUM_FRAME_PIECE)
    return videos_pieces


def main():
    videos = list(get_video_id())
    logger.info('#{} videos need to be processed'.format(len(videos)))

    videos_pieces = split_videos(videos)

    fe = FrameExtractor(verbose=opt['verbose'])
    fe.start(videos_pieces)
    fe.stop()
    logger.info('all done')


if __name__ == "__main__":
    main()


'''bash
python videoprocess/frame_extraction.py --dataname svd-example
'''


