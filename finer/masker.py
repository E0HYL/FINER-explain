#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   masker.py
@Time    :   2022/05/04 13:41:24
Anonymous Submission
@License :   (C)Copyright 2022
@Desc    :   None
'''

# here put the import lib
import re
import time
import numpy as np
import pandas as pd
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from sklearn.metrics import auc
# from scipy.cluster.hierarchy import linkage
from tqdm import tqdm
from tensorflow.keras import Model

from shap import Explanation
from shap.maskers._text import partition_tree, Text
from shap.utils import safe_isinstance, transformers
from shap.benchmark import SequentialMasker, BenchmarkResult

from .feature_reduction import get_top_mask, mask_x_with_background, repeat_seq_mask
from .metric import FidelityMetric, fidelity_auc_metric
from .utils import sparse2dense_with_segment


class FinerSequentialMasker(SequentialMasker):
    def __init__(self, mask_type, sort_order, background, output_fn, *model_args):
        for arg in model_args:
            if isinstance(arg, pd.DataFrame):
                raise Exception("DataFrame arguments dont iterate correctly, pass numpy arrays instead!")

        # FinerSequentialPerturbation.__init__
        self.inner = FinerSequentialPerturbation(
            output_fn, background, sort_order, mask_type
        )
        self.model_args = model_args

    def __call__(self, explanation, name, segment, percent=0.01, **kwargs):
        # FinerSequentialPerturbation.__call__
        return self.inner(name, explanation, segment, *self.model_args, percent=percent, **kwargs)


class FinerSequentialPerturbation():
    def __init__(self, output_fn, background, sort_order, perturbation):
        # self.f = lambda masked, x, index: model.predict(masked)
        self.output_fn = output_fn.predict if isinstance(output_fn, Model) else output_fn # Model
        self.background = background
        self.sort_order = sort_order # positive | negative | absolute
        self.perturbation = perturbation # keep | remove

        # define our sort order
        if self.sort_order == "positive":
            self.sort_order_map = lambda x: np.argsort(x)
        elif self.sort_order == "negative":
            self.sort_order_map = lambda x: np.argsort(-x)
        elif self.sort_order == "absolute":
            self.sort_order_map = lambda x: np.argsort(abs(x))
        else:
            raise ValueError("sort_order must be either \"positive\", \"negative\", or \"absolute\"!")

    def __call__(self, name, explanation, segment, *model_args, percent=0.01, max_masked_percent=1, silent=False, score_class=1, initial_score=None, filter_correct=True):
        # if explainer is already the attributions
        if safe_isinstance(explanation, "numpy.ndarray"):
            attributions = explanation
        elif isinstance(explanation, Explanation):
            attributions = explanation.values
        else:
            raise ValueError("The passed explanation must be either of type numpy.ndarray or shap.Explanation!")

        assert len(attributions) == len(model_args[0]), \
            f"The explanation passed must have the same number of rows as the model_args that were passed: {attributions.shape} vs {model_args[0].shape}"

        pbar = None
        start_time = time.time() 
        svals = []
        kvals = []

        for i, args in enumerate(zip(*model_args)):
            initial_score = self.output_fn([a[np.newaxis, :] for a in args])[:, score_class]
            if filter_correct and initial_score[0] < 0.5: # filter correctly predicted samples
                continue
            _attributions = self.sort_order_map(attributions[i])
            feature_size = np.ceil((segment[i].max() + 1) * max_masked_percent).astype(int)

            increment = max(1, int(feature_size * percent))
            num_values = 0
            sample_kval = []
            _masked_args = []
            for j in range(0, feature_size, increment): 
                k = min(feature_size, j+increment)

                # Mask Generation
                # 1. assume the perturbation is `remove` -> set top-k features in the mask to False
                binary_mask = get_top_mask(_attributions, segment[i], k) # already a 1D vector: no need to aggregate features
                # 2. revert to True if we only consider positive/negative attributions
                if self.sort_order == 'positive':
                    binary_mask[attributions[i] < 0] = True
                elif self.sort_order == 'negative':
                    binary_mask[attributions[i] > 0] = True
                # check if we should break
                current_num_values = np.count_nonzero(binary_mask == False)
                if current_num_values == num_values:
                    break # early exit: no more features are (and will be) masked if we add the `increment`
                else:
                    num_values = current_num_values
                    sample_kval.append(k)
                # 3. invert the binary mask if perturbation is `keep`
                binary_mask = (self.perturbation == 'keep') ^ binary_mask

                sample_args = []
                for _x in args:
                    sample_args.append(mask_x_with_background(_x, binary_mask, self.background))
                _masked_args.append(sample_args)
            
            _masked_args = np.array(_masked_args)
            arg_num = _masked_args.shape[1]
            masked_args = [_masked_args[:, i, ...] for i in range(arg_num)]
            
            masked_out = self.output_fn(*masked_args)
            values = masked_out[:, score_class]
            values = np.concatenate([initial_score, values])

            svals.append(values)
            kvals.append(np.array(sample_kval))

            if pbar is None and time.time() - start_time > 5:
                pbar = tqdm(total=len(model_args[0]), disable=silent, leave=False, desc="SequentialMasker")
                pbar.update(i+1)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # score_values = np.array(svals, dtype=object) # (batch_size, 1+num_increments)
        score_values = svals
        
        xs = np.linspace(0, 1, 100)
        curves = np.zeros((len(score_values), len(xs)))
        for j in range(len(score_values)): # sample scores
            # percent-score (may be different if 1. positive/negative scores are filtered; 2. segment number is different)
            xp = np.linspace(0, 1, len(score_values[j])) 
            yp = score_values[j]
            curves[j,:] = np.interp(xs, xp, yp)
        # ys = curves.mean(0)
        # std = curves.std(0) # , curve_y_std=std
      
        curve_x = xs * max_masked_percent   
        # avg_local_score = FidelityMetric(curve_x).get_score(curves)
        # global_score = fidelity_auc_metric(xs=curve_x, ys=ys)
        # benchmark_result = BenchmarkResult(self.perturbation + " " + self.sort_order, name, curve_x=curve_x, curve_y=ys, value=global_score)
        # benchmark_result.avg_local_score = avg_local_score
        benchmark_result = FidelityMetric(curve_x).form_benchmark_results(curves, explainer_name=name)
        
        return benchmark_result, (kvals, svals)
        
    def plot(self, xs, ys, auc):
        plt.plot(xs, ys, label="AUC %0.4f" % auc)
        plt.legend()
        xlabel = "Percent Unmasked" if self.perturbation == "keep" else "Percent Masked"
        plt.xlabel(xlabel)
        plt.ylabel("Model Output")
        plt.show()


def matrix_text_tokenizer(s, return_offsets_mapping=True, null_token=[], placeholder_len=0): # , segment=None
    out = {'input_ids': null_token, 'offset_mapping': []}
    _dim = len(s) - len(s.strip('['))
    reg_0 = '\[' * _dim
    reg_1 = '\]' * _dim

    row_split = re.findall(f"({reg_0}.*?{reg_1})", s)
    if row_split[0] != '':
        out['input_ids'] = row_split
        if return_offsets_mapping:
            pos = 0
            for row in out['input_ids']:
                out['offset_mapping'].append((pos+placeholder_len, pos+len(row)+placeholder_len))
                pos += len(row) + 2 # len(sep_str) -> ', '

    # if segment is not None:
    #     s_arr = np.array(out['input_ids'])
    #     breakpoint()
    #     out['input_ids'] = [', '.join(s_arr[segment==i]) for i in np.unique(segment)] # sorted segment ids
    #     out['offset_mapping'] =
    
    return out


def get_pseudo_text(x, i=None):
    pseudo_text = str(x.tolist())[1:-1]
    if i is not None:
        pseudo_text = f'{i}_{pseudo_text}'
    return pseudo_text


class MatrixTokenizer():
    def __init__(self, null_token, mask_token_id=-1, segments=None):
        self.mask_token = null_token
        self.mask_token_id = mask_token_id
        self.segments = segments

    def fetch_segment(self, s):  
        if s not in ['', self.mask_token]:
            segment_id = int(s.split('_')[0])
            return self.segments[segment_id]
        else:
            return None

    def __call__(self, s, return_offsets_mapping=True):
        if s not in ['', self.mask_token]:
            segment_id, matrix_text = s.split('_')
            placeholder_len = len(segment_id) + 1 # '_'
            return matrix_text_tokenizer(matrix_text, return_offsets_mapping, self.mask_token, placeholder_len)
        else:
            return {'input_ids': self.mask_token, 'offset_mapping': []}


class PseudoText(Text):
    def __init__(self, tokenizer=None, mask_token=None, output_type="string", segments=None):
        if tokenizer is None:
            if isinstance(mask_token, np.ndarray):
                mask_token = get_pseudo_text(mask_token)
            else:
                assert isinstance(mask_token, str), f"mask_token must be either str or np.ndarray, got {mask_token}"
            tokenizer = MatrixTokenizer(mask_token, segments=segments)

        # super().__init__(tokenizer, mask_token, collapse_mask_token, output_type)
        self.tokenizer = tokenizer
        self.output_type = output_type
        self.collapse_mask_token = False
        self.input_mask_token = mask_token
        self.mask_token = mask_token # could be recomputed later in this function
        self.mask_token_id = mask_token if isinstance(mask_token, int) else None
        self.keep_prefix = 0
        self.keep_suffix = 0
        self.text_data = True
        self.mask_token = mask_token
        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer(self.mask_token)["input_ids"][self.keep_prefix]
        self.fixed_background = self.mask_token_id is None
        self.default_batch_size = 5
        # cache variables
        self._s = None
        self._tokenized_s_full = None
        self._tokenized_s = None
        self._segments_s = None

        self._segmentation = self.tokenizer.segments
        self._segment_ids = None

    def _standardize_mask(self, mask, s):
        mask = super()._standardize_mask(mask, s)
        self._update_s_cache(s)
        mask = sparse2dense_with_segment(mask, self._segmentation, ignore_negative=False)[0]
        return mask

    # def __call__(self, mask, s):
    #     mask = self._standardize_mask(mask, s)
    #     # ...
       
    def _update_s_cache(self, s):
        super()._update_s_cache(s)
        self._segmentation = self.tokenizer.fetch_segment(s)
        if self._segmentation is not None:
            self._segment_ids = np.unique(self._segmentation)

    def clustering(self, s):
        """ Compute the clustering of tokens for the given string.
        """
        self._update_s_cache(s)
        pseudo_tokens = [str(i) for i in self._segment_ids]
        pt = partition_tree(pseudo_tokens, special_tokens=[])
        # use the rescaled size of the clusters as their height since the merge scores are just a
        # heuristic and not scaled well
        pt[:, 2] = pt[:, 3]
        pt[:, 2] /= pt[:, 2].max()

        return pt

        # self._update_s_cache(s)
        # linkage_matrix = linkage(self._segmentation[:, np.newaxis])

        # linkage_matrix[:, 2] = linkage_matrix[:, 3]
        # linkage_matrix[:, 2] /= linkage_matrix[:, 2].max()
        # return linkage_matrix

    def shape(self, s):
        self._update_s_cache(s)
        return (1, len(self._segment_ids))

    def mask_shapes(self, s):
        self._update_s_cache(s)
        return [(len(self._segment_ids), )]

    def invariants(self, s):
        self._update_s_cache(s)
        invariants = np.zeros(len(self._segment_ids), dtype=np.bool_)
        if self.keep_prefix > 0:
            invariants[:self.keep_prefix] = True
        if self.keep_suffix > 0:
            invariants[-self.keep_suffix:] = True
        # mark separator tokens as invariant
        for i, v in enumerate(self._tokenized_s):
            if v == transformers.getattr_silent(self.tokenizer, "sep_token_id"):
                invariants[i] = True
        invariants[self._segment_ids < 0] = True
        return invariants.reshape(1, -1)

    def feature_names(self, s):
        self._update_s_cache(s)
        return [[f'segment #{id}' for id in self._segment_ids]]


def get_basic_segmentation(image):
    seg = np.arange(image.shape[0])
    return repeat_seq_mask(seg, image[..., 0]) # last dimension: channels


class SegmentationDB():
    def __init__(self, image_segments):
        self.segments = image_segments

    def __call__(self, converted_image):
        img_id = self.get_id_from_processed_img(converted_image)
        original_seg = self.segments[img_id] # 1D
        assert len(original_seg.shape) == 1
        new_segment = np.zeros((len(original_seg) + 1, ), dtype=original_seg.dtype)
        new_segment[0] = -2
        new_segment[1:] = original_seg
        new_segment = repeat_seq_mask(new_segment, converted_image[..., 0]) # 2D: as lime_image fuse channels of 3D images with 2D locations
        return new_segment

    def get_id_from_processed_img(self, image):
        return int(image[0, 0, 0])


class LIMEImagePredict():
    def __init__(self, classifier, id_attached=False, original_dim=3):
        self.id_attached = id_attached
        self.original_dim = original_dim
        self.classifier = classifier

    def get_classifier_fn(self):
        return self.predict_converted_img

    def predict_converted_img(self, images):
        _index = int(self.id_attached)
        if self.original_dim == 3:
            images = images[:, _index:, ...]
        elif self.original_dim == 1:
            images = images[:, _index:, 0, 0]
        else: # 2
            images = images[:, _index:, ..., 0]
        return self.classifier.predict(images)


def convert_images(images, ids=None):
    ids = np.arange(len(images)) if ids is None else ids

    new_shape = (images.shape[0], images.shape[1] + 1, ) + images.shape[2:]
    new_images = np.zeros(new_shape)
    for i in range(len(images)):
        new_images[i, 0, ...] = ids[i] # put id in the first row of the image
        new_images[i, 1:, ...] = images[i]
    
    return new_images


def transform_images(images, convert=False, ids=None):
    images = images[..., np.newaxis] if len(images.shape) == 2 else images
    images = convert_images(images, ids=ids) if convert else images
    images = gray2rgb(images) if len(images.shape) == 3 else images
    return images

