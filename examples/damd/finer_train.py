#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   finer_train.py
@Time    :   2022/04/03 09:18:15
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os

# from finer.finer_train import FinerTrainingModel
from finer.finer_update import FINERAnomalyClassifier#, delete_model_embedding

from ._dataset import data_dir, bg_fixed
from .original_train import original_train
from .. import load_trained_model, write_finer_config


model_path = os.path.join(data_dir, 'finer_model')
if not os.path.isdir(model_path):
    os.makedirs(model_path)


# def finer_train(no_epochs=100, batch_size=20, k=1, anomaly_class=1, alpha=1, beta=0, reg_exploss=1e-2, fix_bg=True, bg_backup=bg_fixed, grad_abs=True, segment='class'):
#     original_model = load_trained_model(data_dir, name='original')

#     bg_fixed_value = bg_fixed if fix_bg else None
#     finer_model = FinerTrainingModel(original_model, k=k, anomaly_class=anomaly_class, alpha=alpha, beta=beta, reg_exploss=reg_exploss, bg_fixed=bg_fixed_value, bg_backup=bg_backup, grad_abs=grad_abs)

#     finer_identifier = f'{segment}_{k}_{alpha}_{beta}_{reg_exploss}' if fix_bg else f'{segment}_{k}_{alpha}_{beta}_benign-bg_{reg_exploss}'
#     finer_model_path = os.path.join(model_path, finer_identifier)
#     if not os.path.exists(finer_model_path):
#         os.makedirs(finer_model_path)

#     # if segment == 'class':
#     #     segment_flag = True 
#     # elif segment == 'opcode':
#     #     segment_flag = False
#     if segment == 'opcode' or segment == 'function':
#         segment_flag = segment 
#     else:
#         raise KeyError('Invalid segment. Expected `function`|`opcode`.') # `class`|`opcode`
#     original_train(no_epochs=no_epochs, batch_size=batch_size, output_path=finer_model_path, model=finer_model, segment=segment_flag)


def finer_update(no_epochs=100, 
                 batch_size=20,
                 lr=1e-4,
                 frozen_layers=None,
                 save_best_only=False,
                 segment='function',
                 anomaly_class=1,
                 friendly_filter=True, 
                 mask_option='background', 
                 cutoff_option='topk', 
                 cutoff_value=50, 
                 agg_option='sum',
                 red_param=1., 
                 aug_param=1., 
                 red_loss='CE', 
                 aug_loss='CE',
                 pat_param=1., 
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
    
    finer_identifier = f'[{red_param}_{red_loss}_{aug_param}_{aug_loss}]{friendly_filter}-{mask_option}-{cutoff_option}-{cutoff_value}-{agg_option}'
    finer_model_path = os.path.join(model_path, finer_identifier)
    if not os.path.exists(finer_model_path):
        os.makedirs(finer_model_path)
    write_finer_config(finer_model, finer_model_path)
    
    assert segment in ['function', 'opcode'], 'Invalid segment. Expected `function`|`opcode`.'
    original_train(no_epochs=no_epochs, batch_size=batch_size, output_path=finer_model_path, model=finer_model, segment=segment, lr=lr, save_best_only=save_best_only) # , embedding=True
