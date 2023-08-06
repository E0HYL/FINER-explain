#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_damd.py
@Time    :   2022/04/02 14:14:47
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import unittest
from examples.damd import data_dir, load_dataset, original_train, finer_train, finer_update
from examples import load_trained_model, load_default_background
from finer.explain import local_explain, get_batch_gradients, handle_embedding
from finer.finer_train import FinerTrainingModel
from finer.utils import series_generator_from_numpy, calculate_steps


class TestDAMD(unittest.TestCase):

    # def test_dataset_loader(self):
    #     X_train, X_test, y_train, y_test, s_train, s_test = load_dataset(segment='function')

    #     _bg = load_default_background(data_dir)
    #     if isinstance(_bg, str):
    #         from . import save_background
    #         save_background(_bg, X_train, y_train, num_sample=100)

    #     from finer.utils import series_generator_from_numpy
    #     test_dataset = series_generator_from_numpy(X_test, y_test, batch_size=32, inf=False, segments=s_test)
    #     for data in test_dataset:
    #         for j in data:
    #             print(j.shape)
    #             breakpoint()

    # def test_original_train(self):
    #     original_train(200)

    # def test_explain(self):
    #     _, X_test, _, y_test, _, s_test = load_dataset(segment=True, test_only=True)
    #     test_dataset = series_generator_from_numpy(X_test, y_test, batch_size=8, inf=False, segments=s_test)
    #     for data in test_dataset:
    #         batch_x, batch_y, batch_s = data
    #         break

    #     m = load_trained_model(name='original', system_name='damd')
        
    #     _x, _m = handle_embedding(batch_x, m)
    #     print(_x)
    #     print(_m.summary())
    #     # print(batch_y)
        
    #     g, pred = get_batch_gradients(_x, _m, absolut=True, return_pred=True) # EagerTensor
    #     g = g.numpy()
    #     pred = pred.numpy()
    #     print(g.sum(axis=-1))
    #     assert (pred == m.predict(batch_x)).all()

    #     # g = local_explain('vanilla_gradients', _x, _m, fixed_class=None) # np.ndarray
    #     # print(len(g.sum(axis=-1)[2].nonzero()))

    # def test_finer_train(self):
    #     # finer_train(segment='opcode', k=2500) # mask opcode (embedding layer -> we have already sum the gradients along the last axis)

    #     # finer_train(segment='class', k=4, alpha=1, beta=0, fix_bg=False)
    #     # finer_train(segment='class', alpha=0, beta=1)
    #     # finer_train(segment='class', alpha=1, beta=1, fix_bg=False) 
        
    #     # gpu 3
    #     # finer_train(segment='function', alpha=1, beta=0, k=500, reg_exploss=1)
    #     # gpu 1
    #     # finer_train(segment='opcode', alpha=1, beta=0, k=2500, fix_bg=False, reg_exploss=0.1)
    #     # gpu 2
    #     # finer_train(segment='function', alpha=1, beta=0, k=500, fix_bg=False, reg_exploss=0.1)
    #     # gpu 0 # do not scale
    #     finer_train(segment='function', alpha=1, beta=0, k=10, reg_exploss=0.02, fix_bg=False)

    def test_finer_update(self):
        # GPU 1
        # finer_update(segment='function', red_param=0., aug_param=1., aug_loss='MSE', cutoff_option='topk', cutoff_value=50, agg_option='sum', friendly_filter=False, frozen_layers=[0])
        # # GPU 3
        # finer_update(segment='function', red_param=1e-4, aug_param=0., red_loss='CE', aug_loss='CE', cutoff_option='topk', cutoff_value=50, agg_option='sum')
        # GPU 0
        # finer_update(segment='function', red_param=1e-4, aug_param=3e-5, red_loss='CE', aug_loss='CE', cutoff_option='topk', cutoff_value=50, agg_option='sum')
        # # GPU 0
        # finer_update(segment='function', red_param=0., aug_param=1., aug_loss='MSE', cutoff_option='topk', cutoff_value=50, agg_option='sum')
        finer_update(segment='opcode', red_param=1e-4, aug_param=3e-5, red_loss='CE', aug_loss='CE', cutoff_option='topk', cutoff_value=2500, agg_option='sum')

    # def test_basic_evaluate(self):
    #     m = load_trained_model(name='original')
    #     # breakpoint()
    #     fm = FinerTrainingModel(m)
    #     fm.compile(loss='binary_crossentropy', optimizer=m.optimizer, metrics=m.metrics)

    #     batch_size = 32
    #     _, X_test, _, y_test, _, s_test = load_dataset(segment=True, test_only=True)
    #     steps, _ = calculate_steps(X_test, [], batch_size=batch_size)
    #     test_dataset = series_generator_from_numpy(X_test, y_test, batch_size=batch_size, inf=False, segments=s_test)

    #     fm.evaluate(test_dataset, steps=steps)