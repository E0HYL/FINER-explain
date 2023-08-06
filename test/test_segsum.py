#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_segsum.py
@Time    :   2022/04/20 14:39:21
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import unittest
import numpy as np

from finer.feature_reduction import segment_sum
from time import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def sum_once(e, s):
    return segment_sum(e, s)


def _original_segment_sum(e, seg):
    num_segs = seg.max() + 1# 0 ~ seg.max()
    seg_values = [e[seg == i].sum() for i in range(num_segs)]
    seg_values = np.array(seg_values)  
    return seg_values


class TestSegmentSum(unittest.TestCase):
    def test_segsum(self):
        e_damd = np.random.random([35750,])
        s_damd = np.random.randint(0, 1625, len(e_damd))
        
        e_vul = np.random.random([50, 200])
        s_vul = np.random.randint(0, 8, len(e_vul))
        
        e_dr = np.random.random([20000, 18])
        s_dr = np.random.randint(0, 664, len(e_dr))

        for k, (e, s) in {'DAMD':(e_damd, s_damd), 'VulDeePecker': (e_vul, s_vul), 'DeepReflect': (e_dr, s_dr)}.items():
            print(k)
            T = 0
            T_long = 0
            run_times = 10
            for _ in range(10):
                t0 = time()
                r0 = sum_once(e, s)
                t1 = time()
                T += (t1 - t0)

                t0 = time()
                r1 = _original_segment_sum(e, s)
                t1 = time()
                T_long += (t1 - t0)

                assert r0.all() == r1.all(), f"r0 is\n{r0} and r1 is\n{r1}"

            print('The time is', T/run_times, 'on average while the original implemetation takes', T_long/run_times)

"""
DAMD
The time is 0.0002150297164916992 on average while the original implemetation takes 0.04044516086578369
VulDeePecker
The time is 1.7237663269042968e-05 on average while the original implemetation takes 5.738735198974609e-05
DeepReflect
The time is 0.0006825447082519532 on average while the original implemetation takes 0.03485555648803711
"""

        

