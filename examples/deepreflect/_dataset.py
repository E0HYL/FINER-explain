#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _dataset.py
@Time    :   2022/06/22 14:43:43
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
import numpy as np
from glob import glob
from finer.utils import DataGenerator


data_dir = os.path.join(os.path.dirname(__file__), '_data')
x_y_path = os.path.join(data_dir, 'x_y', '%s.npy')
segment_path = os.path.join(data_dir, 'segments', '%s.npy')
e_path = os.path.join(data_dir, 'e_labels', '%s.npy')
gt_tag = 'malware_gt'
malware_names = ['rbot', 'pegasus', 'carbanak']


def load_dataset_generator(batch_size=64, test_only=False, segment=False, generator=True):
    load_tags = ['test'] if test_only else ['train', 'test']
    dataset_generators = [None] if test_only else []
    for t in load_tags:
        X = np.load(x_y_path % f'X_{t}')
        y = np.load(x_y_path % f'y_{t}')
        s = np.load(segment_path % t) if segment else None
        if generator:
            dataset_generators.append(DataGenerator(X, y, batch_size, s))
        else:
            dataset_generators.append((X, y, s))
    return dataset_generators


def load_concated_ground_truth(malware_names=malware_names):
    X = []
    s = []
    e = []
    for name in malware_names:
        gt_sub_dir_name = f'{gt_tag}/{name}'
        _x = np.load(x_y_path % f'x_{gt_sub_dir_name}') # (1, 20000, 18)
        _s = np.load(segment_path % gt_sub_dir_name) # (seq_len, )
        if len(_s) < _x.shape[1]:
            _s = np.concatenate([_s, [-1 for _ in range(_x.shape[1] - len(_s))]])
        X.append(_x)
        s.append(_s)
        e.append(np.load(e_path % gt_sub_dir_name)) # (seg_num, )
    X = np.concatenate(X)
    return X, np.asarray(s), e


def load_sample_ground_truth(malware_name):
    X = []
    s = []
    e = []
    gt_sub_dir_name = f'{gt_tag}/seperate/{malware_name}*'
    feature_files = glob(x_y_path % f'x_{gt_sub_dir_name}')
    for featureFN in feature_files:
        name = f"{gt_tag}/seperate/{featureFN.split('/')[-1][:-4]}"
        _x = np.load(featureFN)
        _s = np.load(segment_path % name)
        if len(_s) < _x.shape[1]:
            _s = np.concatenate([_s, [-1 for _ in range(_x.shape[1] - len(_s))]])
        X.append(_x)
        s.append(_s)
        e.append(np.load(e_path % name)) # (seg_num, )
    X = np.concatenate(X)
    return X, np.asarray(s), e


def load_ground_truth(malware_names):
    ground_truth = dict()
    for name in malware_names:
        ground_truth[name] = load_sample_ground_truth(name)
    return ground_truth
