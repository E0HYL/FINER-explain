#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _dataset.py
@Time    :   2022/04/07 15:23:36
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
import json
import numpy as np
from gensim.models.word2vec import Word2Vec

from finer.utils import DataGenerator, reset_seg_index, load_basic_seg
from .. import load_seg_by_names


data_dir = os.path.join(os.path.dirname(__file__), '_data')
train_json = os.path.join(data_dir, 'train.json')
test_json = os.path.join(data_dir, 'test.json')
w2v_path = os.path.join(data_dir, 'w2v_model.bin')
token_per_gadget = 50
seq_len = 200
bg_fixed = np.zeros([1, token_per_gadget, seq_len])


def load_from_json(gadgets, w2v, segment=None, **kwargs):
    x = [[w2v.wv[word] for word in gadget["tokens"]] for gadget in gadgets]
    y = [[1,0] if gadget["label"] == 0 else [0,1] for gadget in gadgets]

    types = [gadget["type"] for gadget in gadgets] 
 
    if segment is None:
        s = None
    else:
        if segment == 'token':
            s = load_basic_seg(x)
        elif segment == 'statement':
            indexes = [str(gadget["index"]) for gadget in gadgets]
            t = kwargs.get('t')
            s = load_seg_by_names(data_dir, t, indexes)
        else:
            raise KeyError(f"Invalid `segment`. Expected 'token | statement | None', got {segment}.")
        s = padding(s, types, pad_value=-1)

    x = padding(x, types)

    return np.array(x), np.array(y), s


def pad_one(xi_typei, pad_value=0):
    xi, typei = xi_typei
    if typei == 1:
        if len(xi) > token_per_gadget:
            ret = xi[0:token_per_gadget]
        elif len(xi) < token_per_gadget:
            if pad_value == 0: # gadget -> token_per_gadget * seq_len
                ret = xi + [[pad_value] * len(xi[0])] * (token_per_gadget - len(xi))
            elif pad_value == -1: # seg -> token_per_gadget
                ret = xi.tolist() + [pad_value] * (token_per_gadget - len(xi))
        else:
            ret = xi
    elif typei == 0 or typei == 2: # Trunc/append at the start
        if len(xi) > token_per_gadget: # truncate at the start
            ret = xi[len(xi) - token_per_gadget:]
            if pad_value == -1:
                ret = reset_seg_index(ret, ignore_negative=False) # no negative values here
        elif len(xi) < token_per_gadget:
            if pad_value == 0:
                ret = [[pad_value] * len(xi[0])] * (token_per_gadget - len(xi)) + xi
            elif pad_value == -1:
                ret = [pad_value] * (token_per_gadget - len(xi)) + xi.tolist()
        else:
            ret = xi
    else:
        raise Exception()

    return ret


def padding(x, types, pad_value=0):
    return np.array([pad_one(bar, pad_value=pad_value) for bar in zip(x, types)])


def load_dataset_generator(train_json=train_json, test_json=test_json, w2v_path=w2v_path, batch_size=64, segment=None, test_only=False, generator=True):
    data_paths = {'train': train_json, 'test': test_json}
    if os.path.exists(w2v_path):
        w2v = Word2Vec.load(w2v_path)
    else:
        w2v = train_w2v(data_paths)

    dataset_generators = []
    for t, data_path in data_paths.items():
        if test_only and t == 'train': 
            dataset_generators.append(None)
            continue

        with open(data_path) as f:
            gadgets = json.load(f)
            x, y, s = load_from_json(gadgets, w2v, segment, t=t)
            if generator:
                dataset_generators.append(DataGenerator(x, y, batch_size, s))
            else:
                dataset_generators.append((x, y, s)) # return tuple

    return dataset_generators


def train_w2v(data_paths, vector_size=200, epochs=100, workers=2):
    import contextlib
    import itertools
    with contextlib.ExitStack() as stack:
        f_list = [stack.enter_context(open(dataset)) for dataset in data_paths]
        gadgets = itertools.chain.from_iterable([json.load(f) for f in f_list])

    x = [gadget["tokens"] for gadget in gadgets]

    # Train Word2Vec
    w2v = Word2Vec(x, min_count=1, vector_size=vector_size, epochs=epochs, workers=workers)

    print("Trained Word2Vec embedding with weights of shape:", w2v.wv.vectors.shape)
    w2v.save(w2v_path)
    print("Written model to: {}".format(w2v_path))

    return w2v
