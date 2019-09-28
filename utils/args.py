# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: args.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-25
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import torch
import argparse
import yaml

from datetime import datetime

parser = argparse.ArgumentParser()

'''basic settings'''
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--num-procs', type=int, default=10, help='number of process')
parser.add_argument('--en-local-log', action='store_true', default=False, help='enable local log')
parser.add_argument('--verbose', action='store_true', default=True)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--output-path', type=str, default='log')
parser.add_argument('--configfile', type=str, default='')
parser.add_argument('--dataname', type=str, default='svd')
parser.add_argument('--approach', type=str, default='')
parser.add_argument('--fps', type=int, default=1)
parser.add_argument('--output-period', type=int, default=10000)

args = parser.parse_args()

args.timestamp = datetime.now().strftime('%m-%d-%H-%M-%S')
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
args.logdir = os.path.join(parentddir, args.output_path, '-'.join(['log', args.timestamp]))

if args.debug:
    args.en_local_log = False

if args.en_local_log:
    os.mkdir(args.logdir)

args.device = 'cpu'
if torch.cuda.is_available():
    args.device = 'cuda'

args.path = '/home/jiangqy/'
args.projpath = os.path.join(args.path, 'program', 'SVD-baselines')
args.datapath = os.path.join('/data1/jiangqy/dataset/', args.dataname)
args.configpath = os.path.join(args.projpath, 'config')

args.metadatapath = os.path.join(args.datapath, 'metadata')
args.framepath = os.path.join(args.datapath, 'frames')
args.videopath = os.path.join(args.datapath, 'videos')
args.infopath = os.path.join(args.datapath, 'infos')
args.featurepath = os.path.join(args.datapath, 'features')


def load_config(filepath):
    config = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as fp:
            f = fp.read()
            config = yaml.load(f, Loader=yaml.FullLoader)

    return config

cfg_file = os.path.join(args.configpath, args.configfile + '.yaml')
opt = load_config(cfg_file)

args = vars(args)
for key in args:
    opt[key] = args[key]


def create_path_if_missing(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)


create_path_if_missing(opt['framepath'])
create_path_if_missing(opt['featurepath'])


