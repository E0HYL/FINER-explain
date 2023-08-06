#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   original_train.py
@Time    :   2022/04/07 15:05:42
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional


from finer.utils import train_binary_model
from ._dataset import load_dataset_generator, data_dir, token_per_gadget, seq_len


model_path = os.path.join(data_dir, 'original_model')
if not os.path.isdir(model_path):
    os.makedirs(model_path)


def get_vuldeepecker_rnn(final_activation='softmax'):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=300), input_shape=(token_per_gadget, seq_len)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation=final_activation))
    print(model.summary())
    return model


def original_train(no_epochs=100, batch_size=64, output_path=model_path, model=None, segment=None):
    train_dataset, test_dataset = load_dataset_generator(batch_size=batch_size, segment=segment)
    vuldeepecker_model = get_vuldeepecker_rnn() if model is None else model

    steps = [None, None]
    train_binary_model(vuldeepecker_model, train_dataset, test_dataset, steps=steps, output_path=output_path, num_epochs=no_epochs)
