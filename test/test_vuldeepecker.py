#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_vuldeepecker.py
@Time    :   2022/04/07 16:02:56
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import unittest
from examples import load_trained_model, load_default_background
from examples.vuldeepecker import data_dir, load_dataset_generator, original_train, finer_train
from finer.explain import local_explain#, get_batch_gradients


class TestVulDeePecker(unittest.TestCase):

    def test_dataset_loader(self):
        train_dataset, _ = load_dataset_generator(generator=False)
        
        _bg = load_default_background(data_dir)
        if isinstance(_bg, str):
            from . import save_background
            train_x, train_y, _ = train_dataset
            save_background(_bg, train_x, train_y, num_sample=100)

    # def test_dataset_generator(self):
    #     _, test_generator = load_dataset_generator(batch_size=1, test_only=True, segment='statement')
    #     for i in test_generator:
    #         for j in i:
    #             print(j)
    #         break

    # def test_original_train(self):
    #     original_train()

    # def test_explain(self):
    #     _, test_generator = load_dataset_generator(test_only=True)
    #     for batch_x, batch_y in test_generator:
    #         break

    #     m = load_trained_model(name='original', system_name='vuldeepecker')
        
    #     # g = get_batch_gradients(batch_x, m)
    #     # breakpoint()
    #     g = local_explain('integrated_gradients', batch_x, m)
    #     assert g.shape == batch_x.shape

    #     print(g)

    # def test_finer_train(self):
    #     # finer_train(segment='token', k=10, alpha=1, beta=0)
        
    #     finer_train(segment='statement', k=1, alpha=1, beta=0)

    #     # finer_train(segment='statement', k=1, alpha=0, beta=1)
        
    #     # finer_train(segment='statement', k=1, alpha=1, beta=1)
