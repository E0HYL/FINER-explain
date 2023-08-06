#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_reduction.py
@Time    :   2022/04/09 20:48:27
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import numpy as np
import unittest

from finer.feature_reduction import feature_reduction_simple, feature_reduction_with_seg


class TestReduction(unittest.TestCase):
    def setUp(self):
        self.x = np.random.random((2,3,3))
        self.e = np.random.random((2,3,3))
        print(f"=== Feature importance scores is ===")
        print(self.e)
    
    def test_reduction_simple(self):
        print("Testing simple (Top-1 & Top-2) reduction")
        print(feature_reduction_simple(self.x, self.e, k=3))

    def test_reduction_seg(self):
        s = np.random.randint(0, 2, size=(2,3,))
        print(f'Feature reduction with segment = \n{s}')
        print(feature_reduction_with_seg(self.x, self.e, s, k=1))
