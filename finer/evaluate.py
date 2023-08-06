#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Time    :   2022/04/11 09:21:38
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import defaultdict
from sklearn import metrics

from shap.maskers import Independent #  Text,
# from shap.benchmark import ExplanationError, ComputeTime

from .explain import local_explain, explainer_shap, explainer_lime
from .masker import FinerSequentialMasker, PseudoText, get_pseudo_text, \
    get_basic_segmentation, SegmentationDB, transform_images, LIMEImagePredict
from .feature_reduction import aggregate_attributions, aggregate_segments
from .utils import sparse2dense_with_segment
from .metric import SparsityMetric, form_benchmark_results
from ._benchmark import plt_benchmark


class ExplainableSystem():
    def __init__(self, classifier, explainer, abstraction=True, **kwargs):
        self.classifier = classifier # Model
        # self.x_shape = tuple(self.classifier.input.shape[1:])

        self.explainer_name = explainer # str
        self.abs_flag = kwargs.pop('abs_flag', True)

        # `abstraction`: default to true, applicable to 'LIME', 'Permutation', 'Partition' and 'Random'; 
        self.abstraction = abstraction
        self.explainer_suffix = '' # for misuse cases (without abstraction): will be set to '-'
        self.fx_format = self._get_fx_format()
        self.explainer_tag = f'{self.explainer_name}{self.explainer_suffix}'

        """
            `data`: to construct explainer masker -> mask_token for `text` and background (training) data for `tabular`
            others arguments are passed to `local_explain`
        """
        self.data = kwargs.pop('data', None)
        if explainer in explainer_shap[-3:] and self.data is not None:
            kwargs['background'] = self.data
        self.explainer_options = kwargs

    def _get_fx_format(self):
        if self.explainer_name in explainer_shap[:-2]:
            if self.explainer_name == 'Kernel':
                fx_format = 'tabular'
            else: # For KernelExplainer, text is not expected.
                fx_format = 'text'
                if not self.abstraction: # explainer masker would be reset during explaining
                    self.explainer_suffix = '-'
        elif self.explainer_name in explainer_lime:
            fx_format = 'image' 
            if not self.abstraction:
                self.explainer_suffix = '-'
        else:
            fx_format = None # white-box explainers do not support segmentation
        return fx_format        

    def _get_pred_function(self, X, format, segments=None):
        assert format in [None, 'tabular', 'text', 'image'], "`format` must be None | 'tabular' | 'text' | 'image', got `%s`." % format
        if format is None:
            model_func = self.classifier
        else:            
            X = X[np.newaxis, :] if X.shape == self.x_shape else X # make sure we accept batch data
            _batch_shape = (-1, ) + self.x_shape
            if format == 'tabular':
                # reshape 1D -> perturbation -> reshape original -> prediction
                X = X.reshape((X.shape[0], -1)) # sample_num * flattened features of the sample

                def model_predict(X):
                    X = X.reshape(_batch_shape)
                    return self.classifier.predict(X)
            elif format == 'text':
                X = X.reshape((X.shape[0], X.shape[1], -1)) # sample_num * seq_len * flattened unit features
                X = [get_pseudo_text(x, i) for i, x in enumerate(X)] # pseudo_text batch: a list of strings

                def model_predict(s_list):
                    s_list = [s_list] if isinstance(s_list, str) else s_list
                    # breakpoint()
                    # revert to the original array format
                    X = np.array([eval('[' + s.split('_')[-1] + ']') for s in s_list]).reshape(_batch_shape)
                    return self.classifier.predict(X)
            else:
                original_dim = len(self.x_shape)
                convert = segments is not None
                X = transform_images(X, convert=convert)
                model_predict = LIMEImagePredict(self.classifier, id_attached=convert, original_dim=original_dim).get_classifier_fn() 

            model_func = model_predict

        return X, model_func

    def _get_explanation(self, X, abs_flag=True, feature_aggregator='sum', segments=None, segment_aggregator='sum'):    
        self.x_shape = X.shape[1:]
        _X, model_func = self._get_pred_function(X, self.fx_format, segments=segments)
        _explainer_seg = segments if self.abstraction else np.array([get_basic_segmentation(x) for x in X])
        masker = self._construct_explainer_masker(segments=_explainer_seg)
        num_features = np.array([len(np.unique(s)) for s in _explainer_seg]) if (self.fx_format != 'tabular' and segments is not None) else np.prod(self.x_shape)

        explanation = local_explain(self.explainer_name, _X, model_func, **self.explainer_options, masker=masker, abs_flag=abs_flag, num_features=num_features)
        
        if self.fx_format == 'tabular':
            explanation = explanation.reshape((-1, ) + self.x_shape)
            feature_attributions = self._aggregate_attributions(explanation, feature_aggregator=feature_aggregator)
            feature_attributions = feature_attributions.astype(float)
        elif self.fx_format in ['image', 'text']:
            feature_attributions = sparse2dense_with_segment(explanation, _explainer_seg, disperse=True)
            assert feature_attributions.shape == segments.shape
        else:
            feature_attributions = explanation

        segment_attributions = self._get_segment_attributions(feature_attributions, segment_aggregator=segment_aggregator, segments=segments)\
             if segment_aggregator is not None else None
        
        return feature_attributions, segment_attributions

    def _get_segment_attributions(self, feature_attributions, segment_aggregator, segments):
        return self._aggregate_attributions(feature_attributions, segment_aggregator=segment_aggregator, segments=segments) if segments is not None else feature_attributions

    def get_output_with_explanation(self, X, segments=None):
        y_pred = self.classifier.predict(X)
        _, explanation = self._get_explanation(X, self.abs_flag, segments)
        return y_pred, explanation

    def evaluate_integral_system(self, test_loader, k=50, anomaly_class=1, alpha=1, beta=0, text_mode=True, piecewise_scale=True, reg_exploss=None, bg_fixed=None, bg_backup=None, grad_abs=True):
        from .finer_train import FinerTrainingModel
        finer_arch_classifer = FinerTrainingModel(self.classifier, k=k, anomaly_class=anomaly_class, alpha=alpha, beta=beta, text_mode=text_mode, piecewise_scale=piecewise_scale, reg_exploss=reg_exploss, bg_fixed=bg_fixed, bg_backup=bg_backup, grad_abs=grad_abs)
        
        finer_arch_classifer.compile(optimizer=self.classifier.optimizer, loss=self.classifier.loss, metrics=self.classifier.compiled_metrics._metrics)
        finer_arch_classifer.evaluate(test_loader)
        results = {_m.name: '%.5f' % _m.result().numpy() for _m in finer_arch_classifer.metrics}

        return results

    def evaluate_classifier(self, *args):
        evaluation_output = self.classifier.evaluate(*args) # (X, y) or data_generator
        metric_dict = {}
        for i, m in enumerate(self.classifier.metrics):
            metric_dict[m.name] = evaluation_output[i]
        return metric_dict

    def get_feature_attribution(self, X, s=None, feature_aggregator='sum', abs_flag=True):
        t0 = time()
        feature_attributions, _ = self._get_explanation(X, segments=s, abs_flag=abs_flag, feature_aggregator=feature_aggregator, segment_aggregator=None) # (batch_size, *self.x_shape)
        used_time = time() - t0
        used_time = self.form_time_benchmark(used_time, X.shape) # total time
        return feature_attributions, used_time

    def form_time_benchmark(self, T, X_shape):
        used_time = form_benchmark_results('compute time', self.explainer_tag, T / X_shape[0]) # .value: average time
        setattr(used_time, 'dataset_shape', X_shape) 
        return used_time       

    def evaluate_explainer(self, X, s=None, bg=None, mask_type='remove', ordering='positive', step_fraction=0.1, max_masked_percent=0.2, output_fn=None, *fn_args, gt=None, feature_aggregator='sum', segment_aggregator='sum', abs_flag=True, feature_attributions=None, used_time=None):
        if feature_attributions is None:
            feature_attributions, used_time = self.get_feature_attribution(X, s, feature_aggregator=feature_aggregator, abs_flag=abs_flag)
        else: 
            # warn: if used_time is not set, it will be None
            self.x_shape = feature_attributions.shape[1:]
        segment_attributions = self._get_segment_attributions(feature_attributions, segment_aggregator=segment_aggregator, segments=s)
                    
        # fidelity: proxy evaluation
        segment = np.array(len(feature_attributions) * [np.arange(feature_attributions.shape[1])]) if s is None else s # single value reduction is not meaningful
        output_fn = self.classifier if output_fn is None else output_fn
        fn_args = (X, ) + fn_args # tuple
        benchmark_results, k_mask_pred = self._run_benchmark(feature_attributions, segment, mask_type, ordering, output_fn, *fn_args, step_fraction=step_fraction, max_masked_percent=max_masked_percent, background=bg) # the maxium number of masks for a sample is (1/step_fraction + 1)

        # sparsity on interpretable components
        sparsity_benchmark = SparsityMetric().form_benchmark_results(segment_attributions, explainer_name=self.explainer_tag)  
        sparsity_benchmark.fa = feature_attributions      
        
        # fidelity: ground truth evaluation
        global_roc_data = []
        if gt is not None:
            for i in range(len(X)):
                local_roc_data = self.evaluate_explainer_with_ground_truth(gt[i], segment_attributions[i])
                global_roc_data.append(local_roc_data)
        
        return sparsity_benchmark, benchmark_results, k_mask_pred, global_roc_data, used_time # , feature_attributions

    def evaluate_explainer_with_ground_truth(self, ground_truth, seg_attributions):
        assert len(ground_truth) == len(seg_attributions), \
            f'The length of ground truth is {len(ground_truth)} while the length of feature attributions is {len(seg_attributions)}'
        # -1: some segments fail to get ground truth
        if -1 in ground_truth:
            seg_attributions = seg_attributions[ground_truth != -1]
            ground_truth = ground_truth[ground_truth != -1]

        return seg_attributions, ground_truth 

    def _aggregate_attributions(self, feature_attributions, feature_aggregator='sum', segment_aggregator=None, **kwargs):
        if len(feature_attributions.shape) > 1 + (feature_attributions.shape[0] != self.x_shape[0]):
            seq_len = self.x_shape[0]
            feature_size = np.prod(self.x_shape[1:])
            feature_attributions = feature_attributions.reshape((-1, seq_len, feature_size))
            feature_attributions = aggregate_attributions(feature_attributions, feature_aggregator) # (batch_size, seq_len, )
        if segment_aggregator is not None:
            segments = kwargs.get('segments')
            assert len(segments) == len(feature_attributions)
            feature_attributions = [aggregate_segments(feature_attributions[i], segments[i], segment_aggregator) for i in range(len(segments))] # list: length of sample_num (each element has a length of non-negative segment_num)
        return feature_attributions
    
    def _construct_explainer_masker(self, segments=None):
        if self.fx_format == 'text': # tokenizer, 
            _token = get_pseudo_text(np.zeros((1, np.prod(self.x_shape[1:])))) if len(self.x_shape) > 1 else '[0] '
            mask_token = _token + ', ' if self.data is None else self.data
            text_tokenizer = PseudoText(mask_token=mask_token, segments=segments)
            return text_tokenizer
        elif self.fx_format == 'image': # segmentation function, 
            segmentation_fn = get_basic_segmentation if segments is None else SegmentationDB(segments)
            return segmentation_fn
        elif self.fx_format == 'tabular' and self.explainer_name != 'Kernel': #  , (background) data
            data_dim = np.prod(self.x_shape)
            if self.data is None:
                data = np.zeros((1, data_dim))
                import warnings
                warnings.warn("No background data provided, use zero vectors instead. \
                    Provide (sampled) training data if you can.")
            elif self.data.shape[1] != data_dim:
                data = self.data.reshape((-1, data_dim))
            if self.explainer_name in explainer_shap:
                return Independent(data)
            elif self.explainer_name in explainer_lime:
                return data
        return None

    def _run_benchmark(self, feature_attributions, segment, mask_type, ordering, output_fn, *fn_args, step_fraction=0.01, max_masked_percent=0.5, background=None):
        smasker = FinerSequentialMasker(mask_type, ordering, background, output_fn, *fn_args)
        benchmark_results = smasker(feature_attributions, self.explainer_tag, percent=step_fraction, max_masked_percent=max_masked_percent, segment=segment)
        return benchmark_results


def plot_benchmark(benchmark_results, detailed=False, figname=None, ylim=[0, 1], score_name='value', **kwargs):
    if isinstance(benchmark_results, dict):
    # benchmark_result_dict = {metric_name: benchmark_result_list (for all explainers)}
        plt_object = sum((benchmark_results.values(), []))
        plt_benchmark(plt_object, score_name=score_name, **kwargs)
        if detailed:
            for k in benchmark_results:
                plt_benchmark(benchmark_results[k], score_name=score_name, **kwargs)
    elif isinstance(benchmark_results, list):
        if isinstance(benchmark_results[0], list):
            plt_object = sum((benchmark_results, []))
            plt_benchmark(plt_object, score_name=score_name, **kwargs)
            if detailed:
                for r in benchmark_results:
                    plt_benchmark(r, score_name=score_name, **kwargs)
        else:
            # automatic: if multiple metrics are in the benchmark_results, show overall performance; 
            # else show detailed performance
            show = True if figname is None else False
            plt_benchmark(benchmark_results, show=False, score_name=score_name, **kwargs)
            if ylim is not None:
                plt.gca().set_ylim(ylim)
            if not show:
                plt.savefig(figname)
                plt.close() # or it will keeps adding the previous plotted values to the new figure
            else:
                plt.show()


def plot_kboxes(explainer_results_k, ylim=[0,1], figname=None, \
    fig_title=None, shape='violin', **kwargs):
    k_boxes = defaultdict(list)
    for kvals, pvals in zip(explainer_results_k[0], explainer_results_k[1]):
        k_boxes[0].append(pvals[0])
        for i, k in enumerate(kvals):
            k_boxes[k].append(pvals[i+1])
    
    _x = list(k_boxes)
    _y = [k_boxes[i] for i in k_boxes]

    if shape == 'violin':
        plt.violinplot(_y, positions=_x, showmedians=True, **kwargs)
    else:
        notch = True if shape == 'notch' else False
        plt.boxplot(_y, positions=_x, patch_artist=True, notch=notch, **kwargs)

    if fig_title is not None: 
        plt.title(fig_title)
    ax = plt.gca()
    ax.set_ylim(ylim)
    
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()


def plot_gt_roc_curve(sample_explainer_data, explainer_names, output_dir, sample_names=None, show_auc=True, xylabel=None, thres=0.8, linestyles=None, sort_value=True, xlabs=None, ylim=None, legend_pos=None, linewidth=2, color_inds=None, stackplt=False, hline=True):
    # Colors from LibreOffice
    color = ['008000', 'db7093','004586','ffd320','5f5f5f','bf8040','83caff','314004'\
            ,'ccffcc', 'f5d6e0','cce6ff','fff5cc','e6e6e6','f2e6d9','cce9ff', 'f1fccf'] # light version
    #['8ECFC9', 'FFBE7A', 'FA7F6F', '82B0D2', 'BEB8DC', 'E7DAD2']
    colors = {explainer_names[i]: color[i] for i in range(len(explainer_names))} if color_inds is None \
        else {explainer_names[i]: color[color_inds[i]] for i in range(len(explainer_names))}
    if linestyles is not None:
        linestyles = {explainer_names[i]: linestyles[i] for i in range(len(explainer_names))}

    for sample_id, roc_data in enumerate(sample_explainer_data):
        sample_name = f'Sample #{sample_id}' if sample_names is None else sample_names[sample_id]
        explainer_auc = dict()
        for explainer, (roc_y, roc_score) in zip(explainer_names, roc_data):
            if show_auc:
                # Create ROC data
                fpr, tpr, thresholds = metrics.roc_curve(roc_y, roc_score)
                roc_auc = metrics.auc(fpr, tpr)   
            else: 
                fpr, tpr = roc_y, roc_score # x,y
                roc_auc = roc_score.mean()
            explainer_auc[explainer] = (roc_auc, fpr, tpr)

        _explainers = explainer_auc.items()
        if sort_value: _explainers = sorted(_explainers, key=lambda x: x[1][0], reverse=True)
        for i, (explainer, (roc_auc, fpr, tpr)) in enumerate(_explainers): # large to small
            label = f'{explainer} | AUC: {round(roc_auc, 4)}' if show_auc else f'{explainer}'
            linestyle = '--' if linestyles is None else f'{linestyles[explainer]}'
            color = f'#{colors[explainer]}'
            if stackplt:
                plt.stackplot(fpr, tpr, colors=[color], edgecolor='black', labels=[label], linestyle=linestyle)
            else:
                ls = linestyle.split(',')
                if len(ls) > 1:
                    ls, marker = ls
                else:
                    ls = ls[0]
                    marker = None
                plt.plot(fpr, tpr, linestyle=ls, marker=marker, color=color, label=label, linewidth=linewidth, markersize=6.5)

        if hline:
            # Put a line at TPR 80%
            plt.axhline(y=thres,color='grey',linestyle='dotted')

        x_label, y_label = ['FPR', 'TPR'] if xylabel is None else xylabel
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.title('{0} ROC Curves'.format(sample_name))
        legend_pos = 'lower right' if legend_pos is None else legend_pos
        plt.legend(loc=legend_pos)
        
        if ylim is not None:
            plt.gca().set_ylim(ylim)
        if xlabs is not None:
            plt.gca().set_xticks(xlabs)

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(f'{output_dir}/{sample_name}.png')
            plt.clf()
        else:
            plt.show()
