#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   finer_train.py
@Time    :   2022/04/18 16:33:55
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os

# from finer.finer_train import FinerTrainingModel
from finer.finer_update import FINERAnomalyClassifier

from ._dataset import data_dir
from .original_train import original_train
from .. import load_trained_model, write_finer_config


model_path = os.path.join(data_dir, 'finer_model')
if not os.path.isdir(model_path):
    os.makedirs(model_path)


# def finer_train(no_epochs=10, batch_size=32, k=50, anomaly_class=1, alpha=1, beta=0, reg_exploss=1e-2, bg_fixed=bg_fixed, bg_backup=None, grad_abs=True, segment='statement', text_mode=False):
#     original_model = load_trained_model(data_dir, name='original')
#     finer_model = FinerTrainingModel(original_model, k=k, anomaly_class=anomaly_class, alpha=alpha, beta=beta, reg_exploss=reg_exploss, bg_fixed=bg_fixed, bg_backup=bg_backup, grad_abs=grad_abs)

#     finer_identifier = f'{segment}_{k}_{alpha}_{beta}_{text_mode}'
#     finer_model_path = os.path.join(model_path, finer_identifier)
#     if not os.path.exists(finer_identifier):
#         os.makedirs(finer_model_path)

#     original_train(no_epochs=no_epochs, batch_size=batch_size, output_path=finer_model_path, model=finer_model, segment=segment)


def finer_update(no_epochs=100, 
                 batch_size=20,
                 lr=1e-4,
                 save_best_only=True,
                 initial_value_threshold=0.8883,
                 segment='function',
                 anomaly_class=1, 
                 frozen_layers=None,
                 mask_option='background', 
                 cutoff_option='topk', 
                 cutoff_value=50, 
                 agg_option='sum',
                 red_param=1., 
                 aug_param=1., 
                 red_loss='CE', 
                 aug_loss='CE',
                 pat_param=0., 
                 pat_red=False,
                 pat_aug=True,
                 reg_trainable=False):
    original_model = load_trained_model(data_dir, name='original')
    # original_emb_model = delete_model_embedding(original_model)
    # finer_model = FINERAnomalyClassifier(original_emb_model, anomaly_class, friendly_filter, mask_option, cutoff_option, cutoff_value, agg_option, red_param, aug_param, red_loss, aug_loss)
    finer_model = FINERAnomalyClassifier(
                 original_model, 
                 anomaly_class=anomaly_class, 
                 frozen_layers=frozen_layers,
                 mask_option=mask_option, 
                 cutoff_option=cutoff_option, 
                 cutoff_value=cutoff_value, 
                 agg_option=agg_option,
                 reduction=red_param, 
                 augmentation=aug_param, 
                 red_loss=red_loss, 
                 aug_loss=aug_loss,
                 patch=pat_param,
                 patch_reduction=pat_red,
                 patch_augmentation=pat_aug,
                 reg_trainable=reg_trainable
                 )
    
    finer_identifier = f'{segment}-{reg_trainable}[{red_param}_{red_loss}-{aug_param}_{aug_loss}-{pat_param}_{pat_red}+{pat_aug}]{mask_option}-{cutoff_option}-{cutoff_value}-{agg_option}'
    finer_model_path = os.path.join(model_path, finer_identifier)
    if not os.path.exists(finer_model_path):
        os.makedirs(finer_model_path)
    write_finer_config(finer_model, finer_model_path)
    
    assert segment in ['function', 'block'], 'Invalid segment. Expected `function`|`block`.'
    original_train(no_epochs=no_epochs, batch_size=batch_size, output_path=finer_model_path, model=finer_model, segment=segment, lr=lr, save_best_only=save_best_only, initial_value_threshold=initial_value_threshold) # , embedding=True
