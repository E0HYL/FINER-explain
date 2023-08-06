#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   original_train.py
@Time    :   2022/04/18 16:36:03
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os

from finer.utils import train_binary_model
from ._dataset import load_dataset_generator, data_dir


model_path = os.path.join(data_dir, 'original_model')
if not os.path.isdir(model_path):
    os.makedirs(model_path)


def get_initial_model():
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Conv1D, Input, MaxPooling1D, Flatten
    
    def get_trained_encoder(ae_path, inter_layer="conv1d_10"):
        # Load trained model
        model = load_model(ae_path)
        inter_model = Model(inputs=model.inputs, outputs=model.get_layer(inter_layer).output, name='ae_half')
        inter_model.trainable = False
        return inter_model # output shape: (None, 1250, 2048)

    def get_vgg_classifier(input_shape=(1250, 2048), kernel_size=4, strides=2, class_count=2): # half vgg    
        img_input = Input(shape=(input_shape[0], input_shape[1],))

        # Block 1
        x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                    activation='relu',
                    padding='same',
                    name='block1_conv1')(img_input)
        x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                    activation='relu',
                    padding='same',
                    name='block1_conv2')(x)
        x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

        # Block 2
        x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                    activation='relu',
                    padding='same',
                    name='block2_conv1')(x)
        x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                    activation='relu',
                    padding='same',
                    name='block2_conv2')(x)
        x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(class_count, activation='sigmoid', name='predictions')(x) # binary classification: change softmax to sigmoid

        # inputs = img_input

        # Create model.
        model = Model(img_input, x, name='vgg19_half')

        return model

    def vert_stack_model(modelA, modelB):
        # inputA = Input(input_shape_for_A)
        inputA = modelA.input
        outputA = modelA(inputA)
        outputB = modelB(outputA)

        modelC = Model(inputA, outputB)  

        return modelC
        
    ae = get_trained_encoder()
    vgg = get_vgg_classifier()

    model = vert_stack_model(ae, vgg)

    return model


def original_train(no_epochs=100, batch_size=32, output_path=model_path, model=None, segment=False, lr=1e-3, **kwargs):
    dr_classifier = get_initial_model() if model is None else model
    
    train_dataset, test_dataset = load_dataset_generator(batch_size=batch_size, segment=segment)
    steps = [None, None]
    # breakpoint()

    train_binary_model(dr_classifier, train_dataset, test_dataset, steps=steps, output_path=output_path, num_epochs=no_epochs, lr=lr, **kwargs)
