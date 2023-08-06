#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_deepreflect.py
@Time    :   2022/06/24 17:56:32
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
"""
Limit the memory usage of TensorFlow
"""
from tensorflow.config.experimental import list_physical_devices\
    , set_virtual_device_configuration, VirtualDeviceConfiguration\
    , set_memory_growth
gpus = list_physical_devices(device_type='GPU')
# set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=2000)]) # method 1
set_memory_growth(gpus[0], True) # method 2

import unittest
import pickle
import numpy as np
from sklearn import metrics
from finer.evaluate import ExplainableSystem, plot_gt_roc_curve
from finer.explain import FinerExplainers, explainer_shap
from examples import load_trained_model, load_default_background
from examples.deepreflect import load_dataset_generator, load_ground_truth,\
    malware_names, data_dir

import logging
logger = logging.getLogger('DeepReflect')
logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# logger.addHandler(ch)
explainer_names = ['vanilla_gradients', 'IG', 'DeepLIFT', 'LIME', 'LEMNA', 'Partition'] + ['Ensemble', 'EnsembleW', 'EnsembleB']

class TestDR(unittest.TestCase):
    # def test_dataset_loader(self):
    #     train_dataset, test_dataset = load_dataset_generator(generator=False, segment=True)
    #     _bg = load_default_background(data_dir)
    #     if isinstance(_bg, str):
    #         from . import save_background
    #         train_x, train_y, _ = train_dataset
    #         save_background(_bg, train_x, train_y, num_sample=100)
    #     # gt_dataset = load_ground_truth(malware_names)

    # def test_finer_update(self):
    #     from examples.deepreflect import finer_update
    #     # finer_update(segment='function', red_param=1e-5, aug_param=1e-4, red_loss='CE', aug_loss='CE', cutoff_option='ratio', cutoff_value=0.5, agg_option='sum', pat_param=2e-3, pat_red=False, reg_trainable=True)
    #     finer_update(segment='block', red_param=1e-2, aug_param=1e-3, red_loss='CE', aug_loss='CE', cutoff_option='topk', cutoff_value=2500, agg_option='sum', pat_param=2e-3, pat_red=True, pat_aug=False)

    def test_gt_evaluation(self, lazy=True):
        # lazy = False
        if not lazy:
            self.ensemble()
        model_tags = ['original_', 'finer_/function'] # , 'finer_/block'
        scores = []
        for tag in model_tags:
            scores.append(self.gt_evaluate(tag, _open_file_log=True, lazy=lazy)) #
        import pandas as pd

        def _handle(a,b):
            num_com = len(a[0])
            original = []
            finer = []
            for i in range(num_com):
                print('component #', i)
                ai = [aa[i] for aa in a]
                original.append(np.array(ai).max())
                bi = [bb[i] for bb in b]
                finer.append(np.array(bi).max())
                print(pd.DataFrame([ai, bi]))
            # original = [np.array(original).mean()]
            # finer = [np.array(finer).mean()]
            return original, finer

        original = []
        finer = []
        print('rbot\n')
        _x = _handle(scores[0][:9], scores[1][:9])
        original += _x[0]
        finer += _x[1]
        print('pegasus\n')
        _x = _handle(scores[0][9:-9], scores[1][9:-9])
        original += _x[0]
        finer += _x[1]
        print('carbanak\n')
        _x = _handle(scores[0][-9:], scores[1][-9:])
        original += _x[0]
        finer += _x[1]
        original = np.array(original)
        finer = np.array(finer)
        print([o for o in original])
        print([f for f in finer])
        print((finer-original)/original)
        
    #     # # -- concated features
    #     # X, segments, ground_truth = load_ground_truth(malware_names) # gt_dataset: X, s; e_label
   
    #     # m = load_trained_model(data_dir, name='original')

    #     # for explainer_name in ['vanilla_gradients', 'IG', 'LIME', 'LEMNA', 'Partition', 'integrated_gradients']:
    #     #     logger.info(f'------- {explainer_name} -------')
    #     #     es = ExplainableSystem(classifier=m, explainer=explainer_name)

    #     #     for i, sample_name in enumerate(malware_names):
    #     #         _X = X[i : i+1]
    #     #         _s = segments[i : i+1]
    #     #         _gt = ground_truth[i : i+1]

    #     #         sparsity_benchmark, benchmark_results, k_mask_pred, global_roc_data, used_time = es.evaluate_explainer(_X, _s, gt=_gt)

    #     #         roc_score, roc_label = global_roc_data[0]
    #     #         # Create ROC data
    #     #         fpr, tpr, thresholds = metrics.roc_curve(roc_label, roc_score)
    #     #         roc_auc = metrics.auc(fpr, tpr)   
    #     #         logger.info(f'+ {sample_name}: AUC {roc_auc}')
    #     #         logger.info(f'Sparsity {sparsity_benchmark.value}, \t Fidelity {benchmark_results.value}, \t Efficiency {used_time.value}')   

    def ensemble(self):
        self.ensemble_exp()
        self.ensemble_exp(tag='original_')


    def gt_evaluate(self, model_tag, lazy=True, _open_file_log=False, max_masked_percent=0.025):

        # Get FPR at TPR value
        def get_fpr_at_tpr(target_tpr,tpr,fpr):
            return fpr[np.where(tpr >= target_tpr)[0][0]]
        # Get TPR at FPR value
        def get_tpr_at_fpr(target_fpr,tpr,fpr):
            return tpr[np.where(fpr >= target_fpr)[0][0]]

        _model_tag = model_tag.replace('/', '')
        name, suffix = model_tag.split('_')
        m = load_trained_model(data_dir, name=name, suffix=suffix)
        
        result_dir = 'evaluation_results/GroundTruth/deepreflect'
        _data_dir = os.path.join(result_dir, '_data', _model_tag)
        if not os.path.exists(_data_dir):
            os.makedirs(_data_dir)
        if _open_file_log:
            fh = logging.FileHandler(os.path.join(result_dir, f"{_model_tag}.log"), encoding='utf-8', mode='w')
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)

        gt_dataset = load_ground_truth(malware_names)
        sample_explainer_data = []
        finer_scores = []
        curve_dir = os.path.join(result_dir, 'roc_curves', _model_tag)
        exp_names = []
        if not os.path.exists(curve_dir):
            os.makedirs(curve_dir)
        for malware_name in gt_dataset:
            logger.info(f'+++++++ {malware_name} +++++++')
            X, segments, ground_truth = gt_dataset[malware_name]
            explainer_data = []
            gscores = []
            for explainer_name in explainer_names:
                logger.info(f'------- {explainer_name} -------')
                _data_path = os.path.join(_data_dir, f'{malware_name}-{explainer_name}.pickle')
                if os.path.exists(_data_path) and lazy:
                    with open(_data_path, 'rb') as handle:
                        eval_results = pickle.load(handle)
                else:
                    # fx_format = FinerExplainers().get_fx_format(explainer_name, abstraction=True)
                    # data = load_default_background(data_dir) if fx_format != 'text' else None
                    data = load_default_background(data_dir) if explainer_name in explainer_shap[-3:] else None
                    es = ExplainableSystem(classifier=m, explainer=explainer_name, data=data) 
                    fa_time_path = os.path.join(_data_dir, f'{malware_name}-{explainer_name}.fa.npy')
                    print(fa_time_path)
                    if os.path.exists(fa_time_path):
                        # print(f'loading existing explanation')
                        with open(fa_time_path, 'rb') as handle:
                            fa, tb = pickle.load(handle)  
                    else:
                        fa, tb = es.get_feature_attribution(X, segments)
                        with open(fa_time_path, 'wb') as handle:
                            pickle.dump((fa, tb), handle)
                    deduction_background = load_default_background(data_dir, anomaly=False)
                    try:
                        eval_results = es.evaluate_explainer(X, segments, gt=ground_truth, max_masked_percent=max_masked_percent, feature_attributions=fa, bg=deduction_background)
                    except Exception:
                        print(fa.shape)
                        breakpoint()
                    with open(_data_path, 'wb') as handle:
                        pickle.dump(eval_results, handle)
                sparsity_benchmark, benchmark_results, _, global_roc_data, _ = eval_results
                logger.info(f'Sparsity {sparsity_benchmark.value}, \t Fidelity {benchmark_results.value}') #, \t Efficiency {tb.value}

                aucs = []
                fprs = []
                tprs = []
                target_tpr = 0.9
                target_fpr = 0.1
                for roc_score, roc_label in global_roc_data:
                    # Create ROC data
                    _fpr, _tpr, _thresholds = metrics.roc_curve(roc_label, roc_score)
                    _roc_auc = metrics.auc(_fpr, _tpr)   
                    logger.info(f'+ Local AUC {_roc_auc}')
                    logger.info(f'FPR@{target_tpr}: {get_fpr_at_tpr(target_tpr, _tpr, _fpr)}, TPR@{target_fpr}: {get_tpr_at_fpr(target_fpr, _tpr, _fpr)}')
                    aucs.append(_roc_auc)
                    tprs.append(_tpr)
                    fprs.append(_fpr)
                finer_score = np.array(aucs).max()
                logger.info(f'Avg Local AUC {finer_score}')
                finer_scores.append(np.array(aucs))

                global_scores = np.concatenate([s[0] for s in global_roc_data])
                global_labels = np.concatenate([s[1] for s in global_roc_data])
                explainer_data.append((global_labels, global_scores))
                fpr, tpr, thresholds = metrics.roc_curve(global_labels, global_scores)
                global_score = metrics.auc(fpr, tpr)
                gscores.append(global_score)
                logger.info(f'Global AUC {global_score}')
                logger.info(f'FPR@{target_tpr}: {get_fpr_at_tpr(target_tpr, tpr, fpr)}, TPR@{target_fpr}: {get_tpr_at_fpr(target_fpr, tpr, fpr)}')
            sample_explainer_data.append(explainer_data)
            gscores = np.array(gscores)
            # breakpoint()
            np.save(os.path.join(curve_dir, malware_name+'.npy'), explainer_data[np.where(gscores == gscores.max())[0].astype(int)[0]])
            exp_names.append(explainer_names[np.where(gscores == gscores.max())[0].astype(int)[0]])

        plot_gt_roc_curve([d[:6] for d in sample_explainer_data], ['Gradients', 'IG', 'DeepLIFT', 'LIME', 'LEMNA', 'Shapley'], output_dir=curve_dir, sample_names=malware_names)
        plot_gt_roc_curve([d[6:] for d in sample_explainer_data], ['Ensemble_U', 'Ensemble_W', 'Ensemble_B'], output_dir=curve_dir, sample_names=[m+'_ensemble' for m in malware_names])
        print(exp_names)
        return finer_scores


    def ensemble_exp(self, tag='finer_function', k_per=0.5):
        from run_fidelity_evaluation import sequential_k_masker
        from finer.explain import ensemble_explanation
        
        explanations = []
        weights = []
        _white = [0,1,2]
        _black = [3,4,5]
        background = load_default_background(data_dir, anomaly=False)
        base_dir = f'evaluation_results/GroundTruth/deepreflect/_data/{tag}'
        name, suffix = tag.split('_')
        model = load_trained_model(data_dir, name=name, suffix='/'+suffix)

        gt_dataset = load_ground_truth(malware_names)
        for malware_name in gt_dataset:
            X, segments, ground_truth = gt_dataset[malware_name]
            num_seg = np.array([1 + s.max() for s in segments])
            print(f'{malware_name} #segments:', num_seg)
            # 2. load explanations for candidate explainers
            weights = []
            explanations = []
            for explainer in ['vanilla_gradients', 'IG', 'DeepLIFT', 'LIME', 'LEMNA', 'Partition']: # 'IG'
                attr_path = f'{base_dir}/{malware_name}-{explainer}.fa.npy'
                with open(attr_path, 'rb') as h:
                    attributions = pickle.load(h)[0] 
                if len(attributions.shape) > 2: attributions = attributions.sum(-1)
                da_drop = []
                for ind in range(len(X)):
                    xi = np.array([X[ind]])
                    si = np.array([segments[ind]])
                    ai = np.array([attributions[ind]])
                    ki = num_seg[ind] * k_per
                    _, vs = sequential_k_masker(model, xi, si, ai, background, max_k=ki, k_num=1, duplicate=False)
                    da_drop += [v[0] - v[-1] for v in vs] # np.clip(v[0] - v[-1], 0, None)
                da_drop = np.array(da_drop)
                print(explainer, da_drop.mean())
                weights.append(da_drop)
                explanations.append(attributions)
            # 3. ensemble
            explanations = np.array(explanations) # num_explainers, dataset_size, explanation_shape
            weights = np.array(weights) # num_explainers, dataset_size
            ensembled_explanations = [[],[],[]] 
            for data_id in range(len(X)):
                multiple_explanations = explanations[:, data_id, :]
                exp_weights = weights[:, data_id]
                ensembled_explanations[0].append(ensemble_explanation(multiple_explanations, exp_weights, norm_w=True))
                ensembled_explanations[1].append(ensemble_explanation(multiple_explanations[_white], exp_weights[_white], norm_w=True))
                ensembled_explanations[2].append(ensemble_explanation(multiple_explanations[_black], exp_weights[_black], norm_w=True)) 
            with open(f'{base_dir}/{malware_name}-Ensemble.fa.npy', 'wb') as h:
                pickle.dump((np.array(ensembled_explanations[0]), np.nan), h)
            with open(f'{base_dir}/{malware_name}-EnsembleW.fa.npy', 'wb') as h:
                pickle.dump((np.array(ensembled_explanations[1]), np.nan), h)
            with open(f'{base_dir}/{malware_name}-EnsembleB.fa.npy', 'wb') as h:
                pickle.dump((np.array(ensembled_explanations[2]), np.nan), h)
            
#[0.88646302 0.87782805 0.83902439 0.87613636 0.98138298 0.96031746 0.91304348 0.92307692 0.90566038 0.8358396  0.91304348 0.75675869 0.98275862]