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

        self.input = mp.Queue()

        for idx in range(self.num_procs):
            p = mp.Process(target=self.worker, args=(idx, ))
            p.start()
            self.procs.append(p)

    @staticmethod
    def cv2_extractor(videopath, fps=1):
        try:
            cap = cv2.VideoCapture(videopath)
            cnt = cap.get(cv2.CAP_PROP_FPS)
            count, num_frame = 0, 0
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if isinstance(frame, np.ndarray):
                    if int(count * fps % (round(cnt))) == 0:
                        frames.append(frame)
                        num_frame += 1
                else:
                    break
                count += 1
            cap.release()
            if num_frame > 0:
                return num_frame, frames
        except Exception as e:
            print('Exception: {} (cv2)'.format(e))
            return None

    @staticmethod
    def ffmpeg_extractor(videopath, fps):
        try:
            reader = imageio.get_reader(videopath, 'ffmpeg')
            duration = reader.get_meta_data()['duration']
            num_frame = 0
            frames = []
            for count, im in enumerate(reader):
                image = skimage.img_as_uint(im).astype(np.uint8)
                if int(count * fps % (round(duration))) == 0:
                    frames.append(image)
                    num_frame += 1
            if num_frame > 0:
                return num_frame, frames
        except Exception as e:
            print('Exception: {} (ffmpeg)'.format(e))
            return None

    def frame_extractor(self, params):
        idx, video = params[0], params[1]
        framepath = os.path.join(opt['framepath'], video)
        if os.path.exists(framepath):
            return None
        try:
            os.mkdir(framepath)
            videopath = os.path.join(opt['videopath'], video)
            results = self.cv2_extractor(videopath, self.fps)
            if results is not None:
                num_frame, frames = results
                if self.verbose and idx % opt['output_period'] == 0:
                    logger.info('index: {:6d}. frame extraction for video {} is done by cv2'.format(
                        idx, video))

            else:
                results = self.ffmpeg_extractor(videopath, self.fps)
                if results is not None:
                    num_frame, frames = results
                    if self.verbose and idx % opt['output_period'] == 0:
                        logger.info('index: {:6d}. frame extraction for video {} is done by cv2'.format(
                            idx, video))
                else:
                    logger.info('processing video: {} failed'.format(video))
            for ind, frame in enumerate(frames):
                _framepath = os.path.join(framepath, '{:04d}.jpg'.format(ind))
                cv2.imwrite(_framepath, frame)
            infopath = os.path.join(framepath, 'info.h5')
            fw = h5py.File(infopath, mode='w')
            fw.create_dataset(name='num_frames', data=num_frame)
            fw.close()
        except Exception as e:
            logger.info('Exception: {}, video: {}'.format(e, video))

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

    def start(self, videos):
        for idx, video in enumerate(videos):
            self.input.put([idx, video])
        self.input.put(None)

    def stop(self):
        for idx, proc in enumerate(self.procs):
            proc.join()
            logger.info('process: {} done'.format(idx))


def main():
    videos = list(get_video_id())
    logger.info('#{} videos need to be processed'.format(len(videos)))

    fe = FrameExtractor(verbose=opt['verbose'])
    fe.start(videos)
    fe.stop()
    logger.info('all done')


if __name__ == "__main__":
    main()


'''bash
python videoprocess/frame_extraction.py --dataname svd-example
'''


