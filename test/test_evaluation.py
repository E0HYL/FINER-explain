#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_evaluation.py
@Time    :   2022/04/25 10:44:41
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pickle
import unittest
import logging

from time import time
from examples import load_trained_model, load_default_background
from finer.evaluate import ExplainableSystem, plot_benchmark
from finer.explain import FinerExplainers

system_name = 'damd' # 'vuldeepecker'  'toy_color'


if system_name == 'damd':
    from examples.damd import load_dataset, data_dir, load_segments
    from finer.utils import series_generator_from_numpy
    _, X_test, _, y_test, _, _ = load_dataset(test_only=True)
    seg = 'class'
    s_test = load_segments(seg_name=seg, data=X_test)
    test_dataset = series_generator_from_numpy(X_test, y_test, batch_size=len(X_test), inf=False, segments=s_test)
    for data in test_dataset:
        X_test, y_test, s_test = data
    suffix_original = ''
    if seg == 'class':
        suffix_finer = '/class_1_1_0'
    elif seg == 'opcode':
        suffix_finer = '/opcode_2500_1_0'
    original_tag = 'Original'
    finer_tag = seg
else:
    if system_name == 'toy_color':
        from examples.toy_color import load_dataset_generator, load_ground_truth, data_dir
        seg = True
        model_arch = 'cnn'
        if model_arch == 'cnn':
            suffix_original = '/cnn_2000' # '/mlp_2000'
            suffix_finer = '/cnn_2000_True_5_1_1'  
        elif model_arch == 'mlp':
            suffix_original = '/mlp_2000'
            suffix_finer = '/mlp_2000_True_1_1_0'
        original_tag = f'Original_{model_arch}'
        finer_tag = model_arch
    elif system_name == 'vuldeepecker':
        from examples.vuldeepecker import load_dataset_generator, data_dir
        seg = 'statement' # 'token'
        suffix_original = ''
        if seg == 'statement':
            suffix_finer = '/statement_1_1_1'
        elif seg == 'token':
            suffix_finer = '/token_10_1_0'
        original_tag = 'Original'
        finer_tag = seg
    _, test_dataset = load_dataset_generator(segment=seg, generator=False, test_only=True)
    X_test, y_test, s_test = test_dataset
    
original_classifier = load_trained_model(name='original', system_name=system_name, suffix=suffix_original)
finer_classifier = load_trained_model(name='finer', system_name=system_name, suffix=suffix_finer)

classifier_dict = {original_tag: original_classifier, f'Finer_{finer_tag}': finer_classifier}

filter_anomaly = True # False
benign_bg = False
test_misuse = False

# result directories
base_result_dir = 'evaluation_results'
explainer_fidelity_result = 'FidelityTest'


class TestEvaluation(unittest.TestCase):
    # def test_evaluation(self):
    #     l = MultipleExplainerLayer(classifier=classifier, explainer_list=['vanilla_gradients'])
    #     for x, y, s in test_generator:
    #         fidelity_loss, interpretability_loss, x_reduction_pred, x_prime_pred = l((x,s), classifier, bg_x)
    #         fidelity_loss, interpretability_loss, x_reduction_pred, _ = l((x, None), m, bg_x)

    # def test_basic_evaluation(self):
    #     _, test_generator = load_dataset_generator(test_only=True, segment=seg, batch_size=1)
    #     original_evaluation_obj = ExplainableSystem(classifier=original_classifier, explainer=None)
    #     finer_evaluation_obj = ExplainableSystem(classifier=finer_classifier, explainer=None)

    #     ori_results = original_evaluation_obj.evaluate_integral_system(test_generator)
    #     fin_results = finer_evaluation_obj.evaluate_integral_system(test_generator)

    #     print("Original model:", ori_results)
    #     print("Finer model:", fin_results)
        
    def test_explainer_comparison(self):
        print(system_name, "- Explain Anomaly class:", filter_anomaly)
        X_bg = load_default_background(data_dir, anomaly=False) if benign_bg else None

        for classifier_name in classifier_dict: # classifier
            X = X_test
            y = y_test
            s = s_test
            if filter_anomaly:
                X = X[y.argmax(-1) == 1]
                s = s[y.argmax(-1) == 1]
                y = y[y.argmax(-1) == 1]
            # filter those correctly predicted samples 
            y_pred = classifier_dict[classifier_name].predict(X)
            correct_flag = y_pred.argmax(-1) == y.argmax(-1)
            X = X[correct_flag]
            s = s[correct_flag]
            result_dir = os.path.join(base_result_dir, explainer_fidelity_result, system_name, classifier_name)
            
            test_num = 200
            explainer_results = evaluate_explainer(X, s, X_bg, classifier_name, max_test_num=test_num, explainers=['vanilla_gradients', 'integrated_gradients', 'LIME', 'Random', 'Partition'], result_dir=result_dir, rerun_everywhere=False) # , 'Permutation', 'IG'

            # for X, y, s in test_generator: # batch data
            #     if filter_anomaly:
            #         X = X[y.argmax(-1)==1]
            #     if len(X) == 0:
            #         continue

            #     explainer_results = evaluate_explainer(X, s, X_bg, classifier_name)

            #     break # DEBUG: run one batch

            plot_benchmark(explainer_results, figname=os.path.join(result_dir, f'{test_num}.png'))


def evaluate_explainer(X, s, X_bg, classifier_name, default_mask_token=None, max_test_num=None, result_dir=f'evaluation_results/{system_name}/Finer', explainers=None, rerun_everywhere=False):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    logging.basicConfig(filename=os.path.join(result_dir, 'log.log'), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
    if max_test_num is not None:
        X = X[:max_test_num]
        s = s[:max_test_num]
        # s[0][0] = -1 # debug
    finer_explainers = FinerExplainers()
    explainer_results = []
    if explainers is None:
        explainers = finer_explainers.get_supported_explainers() # ['vanilla_gradients', 'integrated_gradients', 'LIME', 'Permutation', 'Partition']
    elif isinstance(explainers, str):
        explainers = [explainers]
    
    T = {}
    for explainer in explainers:
        print(f'===== {explainer} =====')
        explanation_path = os.path.join(result_dir, f'{explainer}.pickle')
        if os.path.exists(explanation_path) and not rerun_everywhere:
            with open(explanation_path, 'rb') as handle:
                benchmark_results = pickle.load(handle)
                explainer_results.append(benchmark_results)
            continue

        t0 = time()
        for abstraction in [True, False][:test_misuse + 1]:
            fx_format = finer_explainers.get_fx_format(explainer, abstraction)
            if fx_format == 'text':
                data = default_mask_token # will use zero vector later if None
            elif fx_format == 'tabular':
                data = load_default_background(data_dir) # training data
            else:
                data = None # no use   
            
            es = ExplainableSystem(classifier=classifier_dict[classifier_name], explainer=explainer, data=data, abstraction=abstraction)

            benchmark_results, _, _, _ = es.evaluate_explainer(X, s=s, bg=X_bg)
            log_msg(f'[AUC] {classifier_name}, {explainer}: {benchmark_results.value}') # value = auc(curve_x, (np.array(curve_y) - curve_y[0]))
            explainer_results.append(benchmark_results)
            # print(y_mask_pred)
                                    
            if explainer not in finer_explainers.get_abstraction_explainers():
                break # run once (abstraction is meaningless)
        T[explainer] = time() - t0
        log_msg(f'[Time] {explainer}: {T[explainer]}s.')
    
        with open(explanation_path, 'wb') as handle:
            pickle.dump(benchmark_results, handle)
    if len(T): print(T) # else: benchmark_results are all read from pickle files

    return explainer_results


def log_msg(msg):
    logging.info(msg)
    print(msg)
