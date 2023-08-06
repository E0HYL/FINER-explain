#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   original_train.py
@Time    :   2022/04/02 15:23:37
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D #, Dropout


from finer.utils import train_binary_model #, series_generator_from_numpy, calculate_steps
from ._dataset import load_dataset_generator, data_dir #, load_dataset


model_path = os.path.join(data_dir, 'original_model')
if not os.path.isdir(model_path):
    os.makedirs(model_path)

# th  DetectMalware_CNN.lua -useCUDA -gpuid 1 -programLen 8192 -nConvFilters 64 -nEpochs 75 -nSamplingEpochs 5 -nConvLayers 1 -seed 1 -learningRate 1e-3 -nEmbeddingDims 8 -kernelLength 8 -saveModel -saveFileName model_tmp -dataDir ./dataset/ -metaDataFile ./config/metaData_small_test.th7 -maxSequenceLength 8192
def get_damd_cnn(no_tokens=227, final_nonlinearity='softmax'): # 218
    embedding_dimensions = 8
    no_convolutional_filters = 64
    number_of_dense_units = 16
    kernel_size = 8
    no_labels = 2
    model = Sequential()
    model.add(Embedding(input_dim=no_tokens+1, output_dim=embedding_dimensions))
    model.add(Conv1D(filters=no_convolutional_filters, kernel_size=kernel_size, padding='valid', activation='relu'))
    # model.add(Conv1D(filters=no_convolutional_filters, kernel_size=kernel_size, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(number_of_dense_units, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(no_labels, activation=final_nonlinearity))
    print(model.summary())
    return model


def original_train(no_epochs=100, batch_size=32, output_path=model_path, model=None, segment=False, lr=1e-3, **kwargs):
    damd_model = get_damd_cnn() if model is None else model
    
    # data = load_dataset(segment=segment)
    # # train_network_batchwise(data, damd_model, no_epochs=no_epochs)
    # X_train, X_test, y_train, y_test, s_train, s_test = data
    # train_dataset = series_generator_from_numpy(X_train, y_train, batch_size=batch_size, segments=s_train)
    # test_dataset = series_generator_from_numpy(X_test, y_test, batch_size=batch_size, segments=s_test)
    # steps = calculate_steps(y_train, y_test, batch_size)

    train_dataset, test_dataset = load_dataset_generator(batch_size=batch_size, segment=segment)
    steps = [None, None]
    # breakpoint()

    train_binary_model(damd_model, train_dataset, test_dataset, steps=steps, output_path=output_path, num_epochs=no_epochs, lr=lr, **kwargs)


if __name__ == '__main__':
    original_train()