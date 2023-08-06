#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   finer_train.py
@Time    :   2022/04/17 21:48:56
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os

from finer.finer_train import FinerTrainingModel

from ._dataset import data_dir, bg_fixed
from .original_train import original_train
from .. import load_trained_model


model_path = os.path.join(data_dir, 'finer_model')
if not os.path.isdir(model_path):
    os.makedirs(model_path)


def finer_train(no_epochs=100, batch_size=32, k=3, anomaly_class=1, alpha=1, beta=0, reg_exploss=1e-2, reg_dynamic=True, fix_bg=True, bg_backup=bg_fixed, grad_abs=True, segment='statement', piecewise_scale=True):
    original_model = load_trained_model(data_dir, name='original')

    bg_fixed_value = bg_fixed if fix_bg else None
    finer_model = FinerTrainingModel(original_model, k=k, anomaly_class=anomaly_class, alpha=alpha, beta=beta, piecewise_scale=piecewise_scale, reg_exploss=reg_exploss, reg_dynamic=reg_dynamic, bg_fixed=bg_fixed_value, bg_backup=bg_backup, grad_abs=grad_abs)

    finer_identifier = f'{segment}_{k}_{alpha}_{beta}'
    finer_model_path = os.path.join(model_path, finer_identifier)
    if not os.path.exists(finer_model_path):
        os.makedirs(finer_model_path)

    original_train(no_epochs=no_epochs, batch_size=batch_size, output_path=finer_model_path, model=finer_model, segment=segment)