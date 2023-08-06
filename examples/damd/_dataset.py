#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _dataset_large.py
@Time    :   2022/08/31 11:08:15
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
import numpy as np

from finer.utils import load_basic_seg, DataGenerator, series_generator_from_numpy


data_dir = os.path.join(os.path.dirname(__file__), '_data')
train_names = os.path.join(data_dir, 'train.txt')
test_names = os.path.join(data_dir, 'test.txt')
x_train_path = os.path.join(data_dir, 'X_train.npy')
x_test_path = os.path.join(data_dir, 'X_test.npy')
y_train_path = os.path.join(data_dir, 'y_train.npy')
y_test_path = os.path.join(data_dir, 'y_test.npy')
s_train_path = os.path.join(data_dir, 's_train.npy')
s_test_path = os.path.join(data_dir, 's_test.npy')
bg_fixed = 0 # 0: nop


def load_dataset(segment='function', test_only=False):
    if test_only: 
        X_train = y_train = s_train = None
    else: 
        X_train = np.load(x_train_path, allow_pickle=True)
        y_train = np.load(y_train_path)
        s_train = load_segments(t='train', seg_name=segment, data=X_train)
    X_test = np.load(x_test_path, allow_pickle=True)
    y_test = np.load(y_test_path)
    s_test = load_segments('test', seg_name=segment, data=X_test)

    return X_train, X_test, y_train, y_test, s_train, s_test


def load_dataset_generator(batch_size=32, segment='function', test_only=False, generator=True):
    X_train, X_test, y_train, y_test, s_train, s_test = load_dataset(segment=segment, test_only=test_only)
    if generator: 
        return [None, DataGenerator(X_test, y_test, batch_size, s_test, batch_padding=True)] if test_only else [DataGenerator(X_train, y_train, batch_size, s_train, batch_padding=True), DataGenerator(X_test, y_test, batch_size, s_test, batch_padding=True)]
    else:
        def pad_dataset(X_, y_, s_):
            data_generator = series_generator_from_numpy(X_, y_, batch_size=len(X_), inf=False, segments=s_)
            for dataset in data_generator: break
            return dataset
        X_test, y_test, s_test = pad_dataset(X_test, y_test, s_test)
        if test_only: 
            return [None, (X_test, y_test, s_test)]
        else:
            X_train, y_train, s_train = pad_dataset(X_train, y_train, s_train)
            return [(X_train, y_train, s_train), (X_test, y_test, s_test)]
        # return [None, (X_test, y_test, s_test)] if test_only else [(X_train, y_train, s_train), (X_test, y_test, s_test)]


def load_segments(t='test', seg_name='function', data=None):
    if seg_name == 'opcode':
        assert data is not None, 'data must not be None when seg_name is `opcode`'
        segments = np.array(load_basic_seg(data))
    elif seg_name == 'function':
        segments = np.load(s_train_path, allow_pickle=True) if t == 'train' else np.load(s_test_path, allow_pickle=True)
    elif seg_name == False:
        segments = None
    return segments