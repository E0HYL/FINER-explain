#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/04/02 19:56:47
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
import math
import logging
import numpy as np
from tensorflow.keras import metrics, backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint


class FinerModelCheckpoint(ModelCheckpoint):
    def set_model(self, model): # override (tf version: 2.4.1)
        try:
            self.model = model.classifier
            # FinerTrainingModel
            if self.monitor  == 'loss': 
                self.monitor = 'CLoss'
            elif self.monitor == 'val_loss': 
                self.monitor = 'val_CLoss'
            self.filepath = self.filepath.replace('{loss:', '{CLoss:')
            self.filepath = self.filepath.replace('{val_loss:', '{val_CLoss:')
        except AttributeError:
            self.model = model
            # print('Not a FinerTrainingModel, checkpoint for the whole model instead.')
        # Use name matching rather than `isinstance` to avoid circular dependencies.
        if (not self.save_weights_only and
            not self.model._is_graph_network and  # pylint: disable=protected-access
            self.model.__class__.__name__ != 'Sequential'):
            self.save_weights_only = True


def train_binary_model(model, train_data, valid_data, steps, output_path, lr=1e-3, num_epochs=100, class_id=1, record_summary=False, checkpoint_monitor='val_Acc', monitor_mode=None, save_best_only=True, initial_value_threshold=None):
    if lr == 'schedule':
        opt = Adam(learning_rate=0.01)
    else:
        opt = Adam(learning_rate=lr)

    basic_metrics = [metrics.BinaryAccuracy(name='Acc'), metrics.Precision(class_id=class_id, name='Pre'), metrics.Recall(class_id=class_id, name='Rec')]
    # Let's train the model using the Adam optimizer
    model.compile(loss='binary_crossentropy',
        optimizer=opt,
        metrics=basic_metrics) # `class_id` default to `None` in keras, but we care about the anomaly class

    callbacks_list = list()
    if lr == 'schedule':
        base = backend.get_value( model.optimizer.lr )
        def schedule(epoch):
            return base / 10.0**(epoch//2)
        callbacks_list.append(LearningRateScheduler( schedule ))

    model_path = os.path.join(output_path, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # model.build(input_shape=(None, max_len, num_features))
    # model.summary()
    if record_summary:
        with open(os.path.join(output_path, 'model_summary.txt'),'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    if monitor_mode is None:
        mode = 'min' if 'loss' in checkpoint_monitor else 'max'
    else:
        mode = monitor_mode
    checkpoint = FinerModelCheckpoint(os.path.join(model_path, '{epoch:02d}-{val_loss:.4f}-{val_Acc:.4f}-{val_Pre:.4f}-{val_Rec:.4f}.h5'), monitor=checkpoint_monitor, verbose=1, save_best_only=save_best_only, mode=mode, initial_value_threshold=initial_value_threshold)
    callbacks_list.append(checkpoint)

    tb_callback = TensorBoard(log_dir=os.path.join(output_path, 'logs'), write_graph=True)
    callbacks_list.append(tb_callback)

    # Train model
    model.fit(
        train_data,
        steps_per_epoch = steps[0],
        epochs=num_epochs,
        validation_data=valid_data,
        validation_steps = steps[1],
        callbacks=callbacks_list
    )


# padding is to align the sequences in each batch
def series_generator_from_numpy(X, y, batch_size, pad_batch=True, maxlen=None, padding='post', inf=True, segments=None): 
    while True:
        start = 0
        end = batch_size

        while start < len(X):
            x = X[start:end]
            s = None if segments is None else segments[start:end]
            if pad_batch:
                x = pad_sequences(x, maxlen=maxlen, dtype=x[0].dtype, padding=padding)
                if segments is not None:
                    s = pad_sequences(s, maxlen=maxlen, dtype=s[0].dtype, padding=padding, value=-1)
            if segments is None:
                yield x, y[start:end]
            else:
                yield x, y[start:end], s

            start += batch_size
            end += batch_size

        if not inf:
            return


def calculate_steps(_train, _test, batch_size):
    steps = []
    for x in [_train, _test]:
        steps.append(int(math.ceil(1. * len(x) / batch_size)))

    return steps


# simple data generator yielding batches of data from a numpy array
class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size, segments=None, \
        batch_padding=False, maxlen=None, padding='post'):
        self.data = data
        self.labels = labels
        self.segments = segments
        self.batch_size = batch_size
        self.batch_padding = batch_padding
        self.maxlen = maxlen
        self.padding = padding

    def __len__(self):
        length = int(np.floor(len(self.data)/self.batch_size))
        return length if len(self.data)%self.batch_size == 0 else length+1

    def __getitem__(self, idx):
        data_batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
        if self.batch_padding:
            data_batch = pad_sequences(data_batch, maxlen=self.maxlen, dtype=data_batch[0].dtype, padding=self.padding)
        label_batch = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        if self.segments is None:
            return data_batch, label_batch
        else:
            segment_batch = self.segments[idx*self.batch_size:(idx+1)*self.batch_size]
            if self.batch_padding:
                segment_batch = pad_sequences(segment_batch, maxlen=self.maxlen, dtype=segment_batch[0].dtype, padding=self.padding, value=-1)
            return data_batch, label_batch, segment_batch


def generate_segmentation_from_mapping(mapping):
    func_list = list(dict.fromkeys(mapping.values()))
    num_segments = len(func_list)
    segmentation = []
    for k in mapping:
        segmentation.append(func_list.index(mapping[k]))

    return num_segments, segmentation    


def reset_seg_index(seg, ignore_negative=True):
    unique_values = np.unique(seg)
    if ignore_negative:
        unique_values = unique_values[unique_values >= 0]
    unique_values = unique_values.tolist()
    reset_dict = {u: unique_values.index(u) for u in unique_values}
    seg = [reset_dict.get(s, s) for s in seg]
    return np.array(seg)


def sparse2dense_with_segment(sparse_arr, segments, ignore_negative=True, disperse=False): 
    """
    sparse_arr: the length is unique ids in segments (maybe explanation outputs from LIME/SHAP_Partition / masks required by PseuduoText)
    segments: each element indicates the segment id
    """
    dense_arr = []
    sparse_arr = [sparse_arr] if (len(sparse_arr.shape) == 1 and sparse_arr.dtype != object) else sparse_arr
    segments = [segments] if len(segments.shape) == 1 else segments
    for i, (arr, segment) in enumerate(zip(sparse_arr, segments)):
        # for one sample
        _dtype = float if arr.dtype == 'O' else arr.dtype
        pad_arr = np.zeros_like(segment, dtype=_dtype)
        _segment_ids = np.unique(segment)
        segment_ids = _segment_ids[_segment_ids >= 0] if ignore_negative else _segment_ids
        if len(arr) > len(segment_ids):
            assert ignore_negative # should be True since the arr is explanation
            arr = arr[_segment_ids >= 0]
        assert len(arr) == len(segment_ids), f"[{i}th sample] the sparse arr is {arr} while segment ids are {_segment_ids}"
        for index, s_id in enumerate(segment_ids): # do not fetch sparse_arr when segment id is negative 
            if disperse:
                arr = arr.astype(float)
                pad_arr[segment == s_id] = arr[index] / (segment == s_id).sum()  
            else:
                pad_arr[segment == s_id] = arr[index]
        dense_arr.append(pad_arr)
    return np.array(dense_arr)


def load_basic_seg(X): # list of sequences
    return [np.array(range(len(x))) for x in X]


def set_logger(logger, base_level=logging.DEBUG, ch_level=logging.DEBUG, fh_name=None, fh_level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(filename)s@%(lineno)d - %(levelname)s: %(message)s')
    # create logger with 'spam_application'  
    logger.setLevel(base_level)  
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(ch_level)
    # add formatter to the handlers
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch) 
    if fh_name is not None:
        fh_level = base_level if fh_level is None else fh_level
        add_fh(logger, fh_name, fh_level, formatter)
    return logger


def add_fh(logger, fh_name, fh_level=logging.INFO, mode='a'):
    formatter = logging.Formatter('%(asctime)s - %(filename)s@%(lineno)d - %(levelname)s: %(message)s')
    fh = logging.FileHandler(fh_name, mode=mode)
    fh.setLevel(fh_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_model_prob(model, data, target_class):
    if data.dtype=='O':
        score = np.concatenate([model.predict(np.array([test_data])) for test_data in data])[:, target_class]
    else:
        score = model.predict(data)[:, target_class]    
    return score