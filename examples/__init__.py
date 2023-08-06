#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2022/04/12 18:13:15
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
import json
from glob import glob
import numpy as np
import pandas as pd


def load_seg_by_names(data_dir, t, names):
    segment_dir = os.path.join(data_dir, 'segments')
    s = [np.load(os.path.join(segment_dir, t, n+'.npy')) for n in names]

    return s


def load_default_background(data_dir, anomaly=True):
    suffix = '' if anomaly else '_0'
    bg_path = os.path.join(data_dir, f'background{suffix}.npy')
    if os.path.exists(bg_path):
        return np.load(bg_path, allow_pickle=True)
    else:
        return bg_path


def load_trained_model(data_dir=None, name='finer', suffix='', **kwargs):
    """
    name: 'finer' | 'original'; default to 'finer'
    [kwargs]
    choose_metric: 'accuracy' | 'precision' | 'recall' | 'f1_score'; default to 'accuracy'
    system_name: 'damd' | 'vuldeepecker' | 'deepreflect' | (directory name in this folder); default to 'damd'
    """
    if data_dir is not None:
        model_base = os.path.join(data_dir, f'{name}_model' + suffix, 'models')
    else:
        system_name = kwargs.get('system_name', 'damd')
        model_base = os.path.join('examples', system_name, '_data', f'{name}_model' + suffix, 'models')
    model_path = glob(os.path.join(model_base, '*.h5'))

    assert len(model_path) != 0, f'model not found in {model_base}'

    if len(model_path) == 1:
        model_path = model_path[0]
    else:
        choose_metric = kwargs.get('choose_metric', 'accuracy')
        model_path = find_best_model_by_default_name(model_path, choose_metric)
    print(f"Loading trained model from {model_path}.")
    
    from tensorflow.keras.models import load_model
    return load_model(model_path)


def find_best_model_by_default_name(model_paths, metric='accuracy'):
    model_performance = [get_base_name(p, extension=False).split('-')[-3:] for p in model_paths]
    default_metrics = ['accuracy', 'precision', 'recall']
    model_performance = pd.DataFrame(model_performance, columns=default_metrics).astype(float)
    model_performance['path'] = model_paths

    if metric == 'f1_score':
        model_performance['f1_score'] = 2* model_performance['precision'] * model_performance['recall'] / (model_performance['precision'] + model_performance['recall'])
    elif metric not in default_metrics:
        raise ValueError('unknown metric')

    best_model_path = model_performance.loc[model_performance[metric].idxmax()].path

    return best_model_path


def get_base_name(p, extension=False):
    basename = os.path.basename(p)
    if not extension:
        basename = os.path.splitext(basename)[0]
    return basename


def write_finer_config(finer_model, finer_model_path):
    with open(os.path.join(finer_model_path, 'FINERAnomalyClassifier.config'), 'w') as f:
        json.dump(finer_model.get_config(), f, indent=4, cls=NpEncoder)
    

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 8)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
