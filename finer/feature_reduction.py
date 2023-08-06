#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   feature_reduction.py
@Time    :   2022/04/01 13:36:27
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import numpy as np


def feature_reduction_with_seg(x, e, seg, k=50, background=None, return_num=False, feature_aggregator='sum', fix_len=True): # batch data
    if isinstance(k, int):
        k = [k for _ in range(len(x))]
    reduction_results = [_feature_reduction_with_seg(x[i], e[i], seg[i], k[i], background, feature_aggregator, fix_len) for i in range(len(x))]

    x_reduction = np.array([r[0] for r in reduction_results])
    if return_num:
        num_features = np.array([r[1] for r in reduction_results])
        x_reduction = (x_reduction, num_features)

    return x_reduction


def _feature_reduction_with_seg(x, e, seg, k=50, background=None, feature_aggregator='sum', fix_len=True):
    """
    x & e: 1D / 2D / 3D
    seg: 1D
    k: a single value or a list of len(seg)
    background: array (n * x.shape) or a single value
    """
    binary_mask = get_top_mask(e, seg, k, feature_aggregator)
    x_reduction = mask_x_with_background(x, binary_mask, background, fix_len)

    return x_reduction, np.count_nonzero(binary_mask==0)


def get_top_mask(attributions, seg, cutoff_value, feature_aggregator='sum', cutoff_option='topk', index_only=False):
    """
    attributions: 1D / 2D / 3D
    seg: 1D
    k: a single value or a list of len(seg)
    """
    target_shape = attributions.shape
    if len(attributions.shape) != 1:
        attributions = attributions.reshape((target_shape[0], -1))
        attributions = aggregate_attributions(attributions, feature_aggregator) # 1D
    seg_values = segment_sum(attributions, seg)

    if cutoff_option == 'thres':
        seg_index = np.where(seg_values >= cutoff_value*seg_values.max())
    elif cutoff_option in ['topk', 'ratio']:
        ranking = np.argsort(seg_values)
        k = cutoff_value if cutoff_option=='topk' else int(np.ceil(cutoff_value*len(seg_values)))
        seg_index = ranking[-k:]

    if index_only: return seg_index
    binary_mask = np.where(np.isin(seg, seg_index), 0, 1) # important features: 0, irrelevant features: 1
    for i in range(1, len(target_shape)):
        binary_mask = np.repeat(binary_mask[..., np.newaxis], target_shape[i], i)
    
    return binary_mask


def mask_x_with_background(x, binary_mask, background=None, fix_len=True):
    if x.shape != binary_mask.shape:
        assert len(x) == len(binary_mask)
        binary_mask = repeat_seq_mask(binary_mask, x)
    x_reduction = binary_mask * x
    if background is not None and background.shape[0]: # None means background is zeros
        if not hasattr(background, 'shape') or background.shape == (): # single value
            background = np.ones((1, )+x.shape, dtype=x.dtype) * background
        try:
            background_sample = background[np.random.randint(len(background))]
            if not fix_len:
                if len(background_sample) > len(x):
                    background_sample = background_sample[:len(x)]
                elif len(background_sample) < len(x):
                    background_sample = np.concatenate([background_sample, np.zeros((len(x)-len(background_sample), )+x.shape[1:], dtype=x.dtype)])
            assert len(background_sample) == len(x), "Sample and background do not have the same length. Please set `fix_len` to False if acceptable."
        except ValueError:
            breakpoint()
        x_reduction += background_sample * (1-binary_mask)
    return x_reduction 


def feature_reduction_simple(x, e, k=50, background=None, batch_data=True):
    if batch_data:
        ranking = argsort_batch_data(e)
    else:
        ranking = argsort_single_data(e) 
    binary_mask = np.ones(ranking.shape, dtype=bool)
    if batch_data:
        if isinstance(k, int):
            k = [k for _ in range(len(ranking))]
        for i in range(len(ranking)):
            binary_mask[i, ranking[i, -k[i]:]] = False
    else:
        binary_mask[..., ranking[..., -k:]] = False

    binary_mask = binary_mask.reshape(x.shape)
    x_reduction = binary_mask * x
    if background is not None:
        if not hasattr(background, 'shape') or background.shape == (): # single value
            x_shape = x.shape[1:] if batch_data else x.shape
            background = np.ones((1, )+x_shape, dtype=x.dtype) * background
        background_sample = background[np.random.randint(len(background))]
        x_reduction += background_sample * (~binary_mask)

    return x_reduction    


def argsort_single_data(x):
    # index of values sorted in ascending order (from small to large)
    return np.argsort(x, axis=None)


def argsort_batch_data(X):
    """ Example
    Input:
        array([[[0.19323288, 0.25982731, 0.0413671 ],
        [0.38209764, 0.58952603, 0.63913534],
        [0.59832384, 0.14307948, 0.35505654]],

        [[0.64390876, 0.77235446, 0.84502651],
        [0.48009088, 0.21199832, 0.8057657 ],
        [0.19408558, 0.33126266, 0.95482233]]])

    Output:
        array([[2, 7, 0, 1, 8, 3, 4, 6, 5],
        [6, 4, 7, 3, 0, 1, 5, 2, 8]])    
    """
    return np.apply_along_axis(lambda x: argsort_single_data(x.reshape(X.shape[1:])), 1, X.reshape(X.shape[0], -1))


def segment_sum(e, seg):  
    # if len(e.shape) == 2:
    #     e = e.sum(axis=-1)
    assert len(e.shape) == 1, "1D sample explanation expected!"
    seg_values = np.bincount(seg[seg >= 0], weights=e[seg >= 0])   
    
    return seg_values 


def repeat_seq_mask(_m, target): # faster when target sequence is long: target.shape[0] >> np.prod(target.shape[1:])
    for i in range(1, len(target.shape)):
        _m = np.repeat(_m[..., np.newaxis], target.shape[i], i)
    return _m


def repeat_seq_mask_(_m, target):
    new_mask = np.zeros(target.shape, dtype=int)
    for i in range(len(_m)):
        new_mask[i, ...] = _m[i]
    return new_mask


def aggregate_segments(e, seg, segment_aggregator):
    if segment_aggregator == 'sum':
        return segment_sum(e, seg)
    elif segment_aggregator == 'max':
        return [e[seg == i].max() for i in np.unique(seg)]
    elif segment_aggregator == 'min':
        return [e[seg == i].min() for i in np.unique(seg)]


def aggregate_attributions(arr, agg, axis=-1):
    if agg == 'max':
        arr = arr.max(axis)
    elif agg == 'mean':
        arr = arr.mean(axis)
    elif agg == 'sum':
        arr = arr.sum(axis)
    else:
        raise ValueError(f"Unknown aggregator! Expected 'max' or 'mean' or 'sum', got {agg}.")
    return arr
