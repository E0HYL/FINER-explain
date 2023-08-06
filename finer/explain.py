#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   explain.py
@Time    :   2022/04/11 14:57:38
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
from matplotlib.widgets import EllipseSelector
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from lime.lime_image import LimeImageExplainer
from lime.lime_tabular import LimeTabularExplainer
from shap import GradientExplainer, DeepExplainer, KernelExplainer, PermutationExplainer, PartitionExplainer, sample
from shap.explainers import Exact
from shap.explainers.other import Random
from ._lemna import LemnaImageExplainer, LemnaTabularExplainer


explainer_lime = ['LIME', 'LEMNA']
explainer_shap = ['Permutation', 'Partition', 'Random', 'Kernel', 'IG', 'DeepLIFT']
explainer_tags = ['vanilla_gradients', 'integrated_gradients'] + explainer_lime + explainer_shap


class FinerExplainers:
    def get_supported_explainers(self):
        return explainer_tags[:-3]

    def _get_all_explainers(self):
        return explainer_tags

    def get_shap_explainers(self, legacy=False):
        if legacy:
            return explainer_shap
        else:
            return explainer_shap[:3]

    def get_lime_explainers(self):
        """
        Explainers that construct a decision boundary using linear model(s). 
        """
        return explainer_lime

    def get_abstraction_explainers(self, random=False):
        shap_explainers = self.get_shap_explainers() if random else self.get_shap_explainers()[:-1]
        return self.get_lime_explainers() + shap_explainers

    def get_fx_format(self, explainer_name, abstraction=True):
        if explainer_name in self.get_shap_explainers():
            return 'text' if abstraction else 'tabular'
        elif explainer_name in self.get_lime_explainers():
            return 'image' if abstraction else 'tabular'
        else: 
            return None


def _handle_model_for_legacy_explainer(oldModel, batch_shape): 
    newInput = tf.keras.layers.Input(batch_shape=batch_shape)
    # newOutputs = oldModel(newInput)
    # newModel = tf.keras.models.Model(newInput,newOutputs)
    # newModel.set_weights(oldModel.get_weights())
    newModel = tf.keras.Sequential()
    newModel.add(newInput)
    for l in oldModel.layers:
        newModel.add(l)
    return newModel
        

def _handle_vuldeepecker_model_for_lrp(v2model):
    weights = v2model.get_weights()
    def get_vuldeepecker_rnn(final_activation='softmax'):
        def _get_model():
            from keras.layers import Dense, Dropout, LSTM, Bidirectional
            tf.compat.v1.disable_v2_behavior()
            model = tf.keras.Sequential()
            model.add(Bidirectional(LSTM(units=300), input_shape=(50, 200)))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation=final_activation))  
            return model      
        model = _get_model()
        model.set_weights(weights)
        return model
    return get_vuldeepecker_rnn()


def _handle_model_for_lrp(v2model, batch_shape=None):
    weights = v2model.get_weights()
    from tensorflow.keras.layers import Input, GlobalMaxPooling1D, MaxPooling1D, Lambda #Embedding, 
    class GlobalMaxPooling1D_(GlobalMaxPooling1D):
        def __init__(self, pool_size):
            super(GlobalMaxPooling1D_, self).__init__()
            self.pool = MaxPooling1D(pool_size=pool_size)
            self.squeeze = Lambda(lambda s: tf.keras.backend.squeeze(s, axis=1))
        def call(self, inputs):
            # the same as writing GlobalMaxPooling1D.
            max_pool1 = self.pool(inputs)
            return self.squeeze(max_pool1)
    def _get_model(oldModel):
        tf.config.run_functions_eagerly(False)
        # tf.compat.v1.disable_v2_behavior()
        # tf.compat.v1.disable_eager_execution()
        _layers = []
        for _l in oldModel.layers:
            if isinstance(_l, tf.keras.Sequential) or isinstance(_l, tf.keras.Model):
                tmp_layers = _l.layers[1:] if isinstance(_l.layers[0], tf.keras.layers.InputLayer) else _l.layers
                _layers += [l0 for l0 in tmp_layers]
            else:
                _layers.append(_l)
        newModel = tf.keras.Sequential()
        if batch_shape is not None:
            newModel.add(Input(batch_shape=batch_shape))
        for l in _layers:
            if isinstance(l, tf.keras.layers.GlobalMaxPooling1D):
                try:
                    pool_size = newModel.layers[-1].output_shape[1]
                except AttributeError:
                    assert batch_shape is not None, "batch_shape must be provided if the last layer has multiple inbound nodes."
                    _x = np.random.random((1,) + batch_shape[1:])
                    pool_size = newModel(_x).shape[1]
                newModel.add(GlobalMaxPooling1D_(pool_size=pool_size))
            else:
                newModel.add(l)
        return newModel
    model = _get_model(v2model)
    model.set_weights(weights)
    return model


def local_explain(tag, X, m, **kwargs):    
    assert tag in explainer_tags, f'Unknown explainer tag {tag}. Please choose from {explainer_tags}'
    
    if tag == 'vanilla_gradients':
        e = _vanilla_explain(X, m, **kwargs)
    elif tag == 'integrated_gradients':
        batch_size = kwargs.pop('batch_size', 20)
        if batch_size is None:
            e = _ig_explain(X, m, **kwargs)
        else:
            e = [_ig_explain(X[batch_id*batch_size:(batch_id+1)*batch_size], m, **kwargs)\
                 for batch_id in range(int(np.ceil(len(X)/batch_size)))]
            e = np.concatenate(e, axis=0)
    elif tag in explainer_lime + explainer_shap:
        fixed_class = kwargs.get('fixed_class', 1)
        
        if tag in explainer_shap[-3:]:
            # legacy explainers params: model, (background) data -> shap_values: (explanations_class0, explanations_class1, ...)
            background = kwargs.get('background', np.zeros((1,)+X.shape[1:]))
            num_sample_bg = kwargs.get('sample_bg', 10)
            background = sample(background, num_sample_bg)

            def _legacy_explain(model=m, batch_data=X, bg=background):
                if tag == 'IG':
                    explainer = GradientExplainer(model, bg)
                elif tag == 'DeepLIFT':
                    # module 'tensorflow.python.eager.backprop' has no attribute '_record_gradient' -> modify "deep_tf.py"
                    # See https://github.com/slundberg/shap/pull/2355/commits/e6f842573943332d6bcc941f46d3c96dc2ac793f
                    explainer = DeepExplainer(model, bg)
                elif tag == 'Kernel': # slow: perturbation is done on np.prod(x_shape) features
                    explainer = KernelExplainer(model, bg)
                e = explainer.shap_values(batch_data)[fixed_class]
                return e

            if m.input_shape[1] is None:
                # fix embedding layer: no gradients
                _X, _m, embedding_flag = handle_embedding(X, m)
                _X = _X.numpy()
                batch_shape = (None, ) + _X.shape[1:]
                # fix the shape for dynamic input; for DeepExplainer, also rebuild the model with eager execution closed
                _m = _handle_model_for_lrp(_m, batch_shape=batch_shape) if tag == 'DeepLIFT' else _handle_model_for_legacy_explainer(_m, batch_shape=batch_shape) 
                if background.dtype == object:
                    background = tf.keras.preprocessing.sequence.pad_sequences(background, maxlen=X.shape[-1], dtype=X[0].dtype, padding='post')
                    if embedding_flag: 
                        background = handle_embedding(background, m, embedding_only=True).numpy()
                batch_size = kwargs.pop('batch_size', 10)
                e = [_legacy_explain(_m, _X[batch_id*batch_size:(batch_id+1)*batch_size], background)\
                     for batch_id in tqdm(range(int(np.ceil(len(X)/batch_size))))]
                if embedding_flag:
                    e = [np.sum(_e, axis=-1) for _e in e]
                e = np.concatenate(e, axis=0)
            else:
                m = _handle_model_for_lrp(m) if tag == 'DeepLIFT' else m
                e = _legacy_explain(model=m)
        else:
            masker = kwargs.get('masker')
            # params: model, masker -> __call__: Explanation -> .values: (batch_size, explanations, output_dim)
            if tag == 'Permutation':
                num_features = kwargs.pop('num_features')
                min_npermutation = 2
                max_evals = max(min_npermutation*(2 * num_features.max() + 1), 500) # it must be at least 2 * num_features + 1
                explainer = PermutationExplainer(m, masker, max_evals=max_evals) # , max_evals='auto'
            elif tag == 'Partition':
                explainer = PartitionExplainer(m, masker, batch_size=1)
            elif tag == 'Exact':
                explainer = Exact(m, masker)
            elif tag == 'Random':
                explainer = Random(m, masker)
            elif tag in explainer_lime:      
                num_samples = kwargs.get('num_samples', 500)
                num_features = kwargs.pop('num_features')
                e = []
                for i, image in enumerate(X):
                    if callable(masker): # segmentation_fn
                        explainer = LimeImageExplainer(feature_selection='none') if tag == 'LIME'\
                            else LemnaImageExplainer(feature_selection='none') # feature_selection='none': we want explanations for all (segment) features
                        explanation = explainer.explain_instance(image, classifier_fn=m, segmentation_fn=masker, num_samples=num_samples, 
                                                                top_labels=None, labels=(fixed_class, )) # `top_labels` is default to 5, which would make `labels` useless
                    else: # numpy.array -> data
                        explainer = LimeTabularExplainer(training_data=masker, feature_selection='none') if tag == 'LIME'\
                            else LemnaTabularExplainer(training_data=masker, feature_selection='none')
                        explanation = explainer.explain_instance(data_row=image, predict_fn=m, num_samples=num_samples, num_features=num_features,
                                                                top_labels=None, labels=(fixed_class, )) # len(image)
                    _ret_exp = explanation.local_exp[fixed_class]
                    if _ret_exp is None:
                        # cvxpy.error.SolverError: Solver 'OSQP' failed
                        explanation = np.zeros(num_features[i])
                    else:
                        # Column 1: segment id, Column 2: segment attribution
                        explanation = pd.DataFrame(_ret_exp).sort_values(by=0) # sort by segment id
                        if callable(masker): 
                            explanation = explanation[explanation[0] > 0] # ignore the first segment id (0: reset from -2 which encodes the image id)
                        explanation = explanation[1].values
                    e.append(explanation)
                e = np.array(e, dtype=object)
                
            if tag in explainer_shap:
                e = explainer(X).values
                if e.dtype == object:
                    e = np.array([i[..., fixed_class] for i in e], dtype=object)
                else:
                    e = e[..., fixed_class]

    if kwargs.get('abs_flag', True):
        e = np.abs(e)

    return e


def _vanilla_explain(x, m, **kwargs):
    batch_size = kwargs.pop('batch_size', 32)
    explanations = []
    for i in range(0, len(x), batch_size):
        explanations.append(get_batch_gradients(x[i:i + batch_size], m, **kwargs).numpy())
    explanations = np.concatenate(explanations)
    return explanations


def _ig_explain(x, m, **kwargs): # with multiple baselines
    """
    [kwargs]
        `baselines`: array/list of background data (np.ndarray)
        `num_steps`: number of interpolation steps between the baseline and the input
        `batch_data`: boolean, `x` is a batch or a single data point
        `num_max_runs`: maxium number of baselines used to average the IG results
    
    reference:https://keras.io/examples/vision/integrated_gradients/
    """
    baselines = kwargs.pop('baselines', None)
    num_steps = kwargs.pop('num_steps', 64)
    
    num_max_runs = kwargs.pop('num_max_runs', 10)
    if kwargs.pop('batch_data', True):
        sample_x = x[0]
    else:
        sample_x = x
    if baselines is None:
        baselines = np.zeros((1, )+sample_x.shape).astype(sample_x.dtype)
    else:
        baselines = baselines.astype(sample_x.dtype)
    num_max_runs = np.min([num_max_runs, len(baselines)])
    if len(baselines) > num_max_runs:
        rindeices = np.random.choice(len(baselines), num_max_runs, replace=False)
    else:
        rindeices = range(num_max_runs)

    integrated_gradients = []
    for r in range(num_max_runs): # for one baseline
        # Run IG.
        # 1. Do interpolation.
        baseline = baselines[rindeices[r]]
        interpolated_x = [
            baseline + (step / num_steps) * (x - baseline)
            for step in range(num_steps + 1)
        ]
        interpolated_x = np.array(interpolated_x).astype(sample_x.dtype) # shape -> (num_interpolated_x, ) + x.shape

        # 2. TODO: postprocess the interpolated data?

        # 3. Get the gradients
        grads = []
        for i in interpolated_x: # the first interpolated point for the (batch) inputs
            grad = _vanilla_explain(i, m, **kwargs)
            grads.append(grad)
        grads = np.array(grads).astype(grad.dtype)

        # 4. Approximate the integral using the trapezoidal rule
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = np.mean(grads, axis=0)

        # 5. Calculate the integrated gradients
        integrated_grads = (x - baseline) * avg_grads

        # Integrated gradients for all the baselines
        integrated_gradients.append(integrated_grads)

    integrated_gradients = np.array(integrated_gradients)

    return np.mean(integrated_gradients, axis=0)


def get_batch_gradients(inputs, model, **kwargs):
    """
    [kwargs]
        `absolute`: take absolute value of the gradients if `True`.
        `fixed_class`: take prediction probabilities of fixed class(es) for calculating the gradients, can be a int or a 1D/2D list-like object (list, ndarray, tensor) indicting explicit class labels or the implicit prediction reference, use the top classes if set to None.
        `return_pred`: return the prediction results of shape (N, 2) together with the gradients.
    """
    absolute = kwargs.get('absolute', True)
    fixed_class = kwargs.get('fixed_class', 1)
    return_pred = kwargs.get('return_pred', False)

    # Convert to Tensor.
    # if not isinstance(inputs, tf.Tensor): # or inputs.dtype.isfloating()
    dtype = model.input.dtype
    inputs = tf.cast(inputs, dtype=dtype)
    inputs, model, has_embedding = handle_embedding(inputs, model)

    # Use GradientTape to get the gradients w.r.t. the inputs.
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds = model(inputs)
        top_class_pred = get_top_class_pred(preds, fixed_class=fixed_class)

    grads = tape.gradient(top_class_pred, inputs)
    if absolute:
        grads = tf.abs(grads)
    if has_embedding:
        grads = tf.reduce_sum(grads, axis=-1)
    
    grads = (grads, preds) if return_pred else grads
    return grads


def ensemble_explanation(exps, weights, norm_w=False):
    # 1. normalize
    for i in range(len(exps)):
        data = exps[i]
        exps[i] = (data - np.min(data)) / (np.max(data) - np.min(data))\
                if np.max(data) - np.min(data) else data
    # 2. weighted sum
    if norm_w:
        # weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) - np.min(weights) else weights
        _weights = np.clip(weights, 0, None)
        if _weights.sum() > 0:
            weights = _weights /_weights.sum()
            weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) 
        else: 
            # breakpoint()
            _weights[np.where(weights == weights.max())] = 1.
            weights = _weights
        print("weights", weights)
    return weights.dot(exps)


def get_top_class_pred(y_pred, fixed_class=None):
    if isinstance(fixed_class, int): # int
        top_class_pred = y_pred[:, fixed_class]
    else:
        if fixed_class is None: # use top prediction classes
            pred_labels = tf.argmax(y_pred, axis=1, output_type=tf.int32)
        elif len(fixed_class.shape) == 2: # reference (binary) probabilities: 2D np.ndarray/tf.Tensor
            pred_labels = tf.argmax(fixed_class, axis=1, output_type=tf.int32)
        else: # explicit labels: 1D list/np.ndarray/tf.Tensor
            pred_labels = fixed_class
        indices = tf.transpose([tf.range(len(y_pred)), pred_labels])
        top_class_pred = tf.gather_nd(y_pred, indices)
    return top_class_pred


def handle_embedding(x, m, embedding_only=False):
    def _get_embedding(_x):
        return m.layers[0](_x)

    if embedding_only:
        return _get_embedding(x)

    if _is_embedding(m.layers[0]):
        _x = _get_embedding(x)
        diff_model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=_x.shape[1:])] + m.layers[1:]) # tf.keras.Model(inputs=m.layers[1].input,outputs=m.outputs)
        return _x, diff_model, True      
    else:
        return x, m, False


def _is_embedding(l):
    return isinstance(l, tf.keras.layers.Embedding)
