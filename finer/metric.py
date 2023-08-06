#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metric.py
@Time    :   2022/06/08 11:09:14
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import numpy as np
from sklearn.metrics import mean_squared_error, auc
from shap.benchmark import BenchmarkResult


class FinerMetric:
    def __init__(self, metric_name, interpolate_auc=False, curve_y_post_fn=None, dataset_desc='dataset_values'):
        self.metric_name = metric_name
        self.interpolate = interpolate_auc
        self.post_fn = curve_y_post_fn # adjust the curve_y values before AUC calculation
        self.dataset_desc = dataset_desc

    def local_metric(self, local_values):
        """Returns 
        local_curve_x
            - Fidelity: the number/percent (`k`) of removed features
            - Sparsity: the interval size (`r`) for calculating the integral in MAZ
        local_curve_y: 
            - Fidelity (w.r.t. `k`)
                - DA (descriptive accuracy): the lower, the better
                - CD / CR (confidence drop/ratio): the higher, the better
            - Sparsity (w.r.t. `r`)
                - MAZ: the higher, the better
        """
        return NotImplementedError

    def global_metric(self, global_values):
        """Returns
        global_curve_x
            - Fidelity: the number/percent (`k`) of removed features
            - Sparsity: the interval size (`r`) for calculating the integral in MAZ
        global_curve_y: 
            - Fidelity (w.r.t. `k`): average of local_curve_y from all samples
            - Sparsity (w.r.t. `r`): MAZ calculated on `global_values`
        """
        return NotImplementedError

    def get_local_score(self, local_curve_x, local_curve_y):
        if self.interpolate:
            xs = np.linspace(0, 1, 100)
            local_curve_y = np.interp(xs, local_curve_x, local_curve_y)
            local_curve_x = xs
        if self.post_fn is not None:
            local_curve_y = self.post_fn(local_curve_y)
        return auc(local_curve_x, local_curve_y)

    def get_score(self, dataset_values):
        scores = []
        for local_values in dataset_values:
            local_curve_x, local_curve_y = self.local_metric(local_values)
            scores.append(self.get_local_score(local_curve_x, local_curve_y))
        return np.array(scores).mean()

    def get_global_score(self, dataset_values, explainer_name=None, return_benchmark=True):
        global_curve_x, global_curve_y = self.global_metric(dataset_values)
        global_score = self.get_local_score(global_curve_x, global_curve_y)
        if return_benchmark:
            benchmark_results = form_benchmark_results(self.metric_name, explainer_name, value=global_score, curve_x=global_curve_x, curve_y=global_curve_y) 
            return benchmark_results
        return global_score

    def form_benchmark_results(self, dataset_values, explainer_name):
        global_benchmark = self.get_global_score(dataset_values, explainer_name=explainer_name)
        benchmark_results = global_benchmark
        # add new attributions
        # 1. dataset_values that are used to calculate all scores from
        setattr(benchmark_results, self.dataset_desc, dataset_values)
        # 2. FINER: measure the metric locally and average at last
        benchmark_results.finer_score = self.get_score(dataset_values)
        return benchmark_results


class SparsityMetric(FinerMetric):
    def __init__(self, num_bins=100, metric_name='Sparsity (MAZ)', interpolate_auc=False, curve_y_post_fn=None):
        self.bins = np.array(range(num_bins + 1)) / num_bins
        super().__init__(metric_name=metric_name, interpolate_auc=interpolate_auc, curve_y_post_fn=curve_y_post_fn, dataset_desc='attributions')

    def _min_max_norm(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))\
                if np.max(data) - np.min(data) else data

    def _maz(self, attributions): # for one sample
        normed_attributions = self._min_max_norm(attributions) # to range=(0,1)
        counts, bins = np.histogram(normed_attributions, bins=self.bins, density=True) # density=True: normalize the counts to [0,1]
        dx = bins[1] - bins[0]
        maz_values = np.concatenate(([0], np.cumsum(counts) * dx))
        # maz_auc = auc(bin_values, maz_values)
        return bins, maz_values # maz_auc

    def local_metric(self, local_attributions):
        if len(local_attributions.shape) > 1:
            local_attributions = local_attributions.flatten()
        return self._maz(local_attributions)

    def global_metric_0(self, global_attributions):
        # local attributions can have different lengths
        global_attributions = np.concatenate(global_attributions)
        return self.local_metric(global_attributions)

    def global_metric(self, global_attributions):
        global_curve_y = []
        for i in global_attributions:
            local_curve_x, local_curve_y = self.local_metric(i)
            assert (local_curve_x == self.bins).all()
            global_curve_y.append(local_curve_y)
        global_curve_y = np.array(global_curve_y).mean(0)
        return self.bins, global_curve_y        


class FidelityMetric(FinerMetric):   
    def __init__(self, deduction_indices, metric_name='DA', interpolate_auc=False, curve_y_post_fn='default'):
        self.deduction_indices = deduction_indices # the first is zero (means that no feature is removed)
        self._metric_name = metric_name
        self.invert = True if metric_name in ['DA', 'PCR'] else False # True: the lower the metric, the higher the fidelity
        metric_name = f'Fidelity ({metric_name})'
        if curve_y_post_fn == 'default':
            # when metric is 'CR', scale the original value from [0, +inf) to [0, 1)
            post_fn = self._cr_post_fn if self._metric_name == 'CR' else None
        else:
            post_fn = curve_y_post_fn
        super().__init__(metric_name, interpolate_auc, post_fn, dataset_desc='deduction_preds')

    def local_metric(self, deduction_preds): # the first element: the original prediction of the sample
        # the lower, the better
        if self._metric_name == 'DA':
            local_curve_y = deduction_preds
        # the higher, the better
        elif self._metric_name == 'CD': 
            original_pred = deduction_preds[0]
            local_curve_y = (original_pred - deduction_preds).clip(min=0)
        elif self._metric_name == 'CR':
            original_pred = deduction_preds[0]
            local_curve_y = (original_pred / deduction_preds).clip(min=1) - 1 # [0, +inf)
            if self.cr_post_fn is not None:
                local_curve_y = self.cr_post_fn(local_curve_y)
        return self.deduction_indices, local_curve_y

    def global_metric(self, deduction_preds):
        global_curve_y = []
        for i in deduction_preds:
            local_curve_x, local_curve_y = self.local_metric(i)
            assert (local_curve_x == self.deduction_indices).all()
            global_curve_y.append(local_curve_y)
        global_curve_y = np.array(global_curve_y).mean(0)
        return self.deduction_indices, global_curve_y

    def get_local_score(self, local_curve_x, local_curve_y): # called when self._metric_name in ['DA', 'CD', 'CR']
        auc_value = super().get_local_score(local_curve_x, local_curve_y)
        return fidelity_auc_metric(auc_value=auc_value, xs=local_curve_x, ys=local_curve_y, invert=self.invert)

    def get_score(self, dataset_deduction_preds): 
        """
        shape of the args:
        - dataset_deduction_preds: ( num_samples, len(self.deduction_indices) )
        """
        if self._metric_name in ['DA', 'CD', 'CR']:
            return super().get_score(dataset_deduction_preds)
        elif self._metric_name == 'PCR':
            pcr_values = []
            for col_index in range(dataset_deduction_preds.shape[1]):
                k_values = dataset_deduction_preds[:, col_index]
                k_pcr = (k_values >= 0.5).sum() / dataset_deduction_preds.shape[0]
                pcr_values.append(k_pcr)
            return self.get_global_score(self.deduction_indices, pcr_values)

    def _cr_post_fn(value, steepness=10): # use S-like function to scale CR to [0, 1)
        return 2 / (1 + np.exp(-steepness * value)) - 1


class DeductionFidelityMetric: # legacy method: global evaluation
    """
    support binary classification
    the 1D prediction vectors passed here represent the probabilities of a specific class (for all samples)
    pcr, ada, rmse: the lower, the better
    """
    def __init__(self, deduction_pred, original_pred=None, approximate_pred=None):
        self.num_samples = len(deduction_pred)
        self.deduction_pred = deduction_pred

        self.original_pred = original_pred
        self.approximate_pred = approximate_pred

    def ada(self):
        """
        Average Descriptive Accuracy in Eval_DL_Sec
        """
        return self.deduction_pred.mean()

    def pcr(self):
        """
        Positive Classification Rate in LEMNA
        """
        return (self.deduction_pred >= 0.5).sum() / self.num_samples

    def rmse(self):
        """
        RMSE for explaination methods that build an approximated decision boundary
        """
        return mean_squared_error(self.original_pred, self.approximate_pred)

    def confidence_change(self, method='minus'):
        return self._quantitative_change(self.original_pred, self.deduction_pred, method).mean()

    def _quantitative_change(self, before, after, method):
        if method == 'minus':
            return before - after
        elif method == 'divide':
            return before / after


def form_benchmark_results(metric_name, explainer_name, value, curve_x=None, curve_y=None, value_sign=1):
    return BenchmarkResult(metric_name, explainer_name, value=value, curve_x=curve_x, curve_y=curve_y, value_sign=value_sign) 


def fidelity_auc_metric(auc_value=None, xs=None, ys=None, thr=None, p0=None, invert=True):
    if (auc_value is None) or (thr is None) or (p0 is None):
        assert (xs is not None) and (ys is not None), \
            'Values along x- and y-axis must be provided if `auc_value` or `thr` or `p0` is None.'
    if auc_value is None:
        auc_value = auc(xs, ys) 
    thr = xs.max() if thr is None else thr
    p0 = ys[0] if p0 is None else p0
    denominator = thr * p0
    numerator = denominator - auc_value if invert else auc_value
    return numerator / denominator