# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: logger.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-25
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import logging
import os

from logging import handlers

from utils.args import opt


class Logger(object):
    def __init__(self,
                 fmt='[%(asctime)s][%(levelname)s]: %(message)s'
                 ):
        self.logger = logging.getLogger()
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        self.logger.addHandler(sh)

        if opt['en_local_log']:
            logfile = os.path.join(opt['logdir'], 'log.log')
            th = handlers.TimedRotatingFileHandler(filename=logfile,
                                                   when='D',
                                                   backupCount=3)
            th.setFormatter(format_str)
            self.logger.addHandler(th)

    def info(self, info):
        self.logger.info(info)

    def debug(self, info):
        self.logger.debug(info)

    def warning(self, info):
        self.logger.warning(info)

    def exception(self, info):
        self.logger.exception(info)


logger = Logger()
for key in opt:
    logger.info('param: {}: {}'.format(key, opt[key]))

logger.info('---------------------------------load param done---------------------------------')

if opt['en_local_log']:
    import yaml
    paramfile = os.path.join(opt['logdir'], 'param.yaml')
    with open(paramfile, 'w') as fp:
        yaml.dump(opt, fp)


