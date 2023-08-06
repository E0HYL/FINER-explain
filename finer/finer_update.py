import tensorflow as tf

tf.config.run_functions_eagerly(True)
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

from tensorflow.keras import Model, metrics
from .explain import get_batch_gradients
from .feature_reduction import get_top_mask, mask_x_with_background


classifier_loss_tracker = metrics.Mean(name='CLoss')
red_loss_tracker = metrics.Mean(name="ELoss_r")
aug_loss_tracker = metrics.Mean(name="ELoss_a")
pat_loss_tracker = metrics.Mean(name="ELoss_p")
explainer_loss_tracker = metrics.Mean(name="ELoss")
total_loss_tracker = metrics.Mean(name="Loss")


class FINERAnomalyClassifier(Model):
    def __init__(self,
                 classifier, 
                 anomaly_class=1,
                 frozen_layers=None,
                 mask_option='background', 
                 cutoff_option='topk', 
                 cutoff_value=50, 
                 agg_option='sum',
                 reduction=1., 
                 augmentation=1., 
                 red_loss='CE', 
                 aug_loss='CE',
                 patch=0.,
                 **kwargs):
        super(FINERAnomalyClassifier, self).__init__(name='FINERAnomalyClassifier')
        self.classifier = classifier
        self.frozen_layers = frozen_layers
        if frozen_layers is not None:
            self._freeze_layers()

        # Sample Division Options
        self.anomaly_class = anomaly_class
        # filter samples where we want to calculate `explainer_loss`

        # Sample Generation Options
        # 1. how to aggregate feature importance into segment attributions
        assert agg_option in ['sum', 'max', 'mean']
        self.agg_option = agg_option
        # 2. how to select important segments
        assert (cutoff_option == 'topk' and isinstance(cutoff_value, int)) or (cutoff_option in ['ratio', 'thres'] and 0. <= cutoff_value <= 1.)
        self.cutoff_option = cutoff_option
        self.cutoff_value = cutoff_value
        # 3. how to replace the segments of interest
        if mask_option == 'constant':
            self.mask_value = kwargs.pop('mask_value', 0)
        else:
            assert mask_option == 'background'
            # mask_value is set during sample dividing (for a batch)
        self.mask_option = mask_option

        # Loss Function Options
        self.reg_trainable = kwargs.pop('reg_trainable', False)
        self.reduction = tf.Variable(float(reduction), trainable=self.reg_trainable, name='R_r', constraint=tf.keras.constraints.NonNeg)
        self.augmentation = tf.Variable(float(augmentation), trainable=self.reg_trainable, name='R_a', constraint=tf.keras.constraints.NonNeg)
        if reduction or patch:
            self.red_loss = red_loss
        if augmentation:
            self.aug_loss = aug_loss
        self.patch = tf.Variable(float(patch), trainable=self.reg_trainable, name='R_p', constraint=tf.keras.constraints.NonNeg)
        if patch:
            self.patch_reduction = kwargs.pop('patch_reduction', False)
            self.patch_augmentation = kwargs.pop('patch_augmentation', True)

    def _freeze_layers(self):
        for i in self.frozen_layers:
            self.classifier.layers[i].trainable = False

    def _filter_sample(self, X, y_pred, y_true):
        """
        to divide batch samples `X` into `X0` and `X1`
        if mask_option == 'background':
            set `mask_value` with `X0`: true negative samples
        return `filter_flag` for `X1`: true positive samples

        """
        anomaly_flag = y_true[:, self.anomaly_class] == 1
        correct_flag = tf.where(y_pred[y_true==1] >= 0.5, True, False)
        if self.mask_option == 'background':
            X0 = X[(~anomaly_flag) & correct_flag]
            self.mask_value = X0
        tp_filter = (anomaly_flag & correct_flag)
        fp_filter = ((~anomaly_flag) & (~correct_flag)) if self.patch else None # prevent low precision
        fn_filter = (anomaly_flag & (~correct_flag))
        return tp_filter, fp_filter, fn_filter

    def _generate_sample_mask(self, X, seg, grads, filter_flag):
        X_s = X[filter_flag]
        G_s = grads[filter_flag]
        S_s = seg[filter_flag]
        M = [get_top_mask(G_s[i].numpy(), S_s[i].numpy(), self.cutoff_value, self.agg_option, self.cutoff_option) \
            for i in range(len(X_s))]
        return X_s, M

    def _generate_meaningful_samples(self, X, seg, grads, filter_flag, red_flag, aug_flag):
        # numpy operations are supported since gradients are not recorded here
        X1, M = self._generate_sample_mask(X, seg, grads, filter_flag)
        if len(X1) == 0:
            X_red = X_aug = tf.convert_to_tensor(()) # None
        else:
            X_red = [mask_x_with_background(X1[i].numpy(), M[i], self.mask_value) \
                for i in range(len(X1))] if red_flag else tf.convert_to_tensor(())
            X_aug = [mask_x_with_background(X1[i].numpy(), 1-M[i], self.mask_value) \
                for i in range(len(X1))] if aug_flag else tf.convert_to_tensor(())
            self.X1 = X1 # _augmentation_mse_loss
        X_red = tf.convert_to_tensor(X_red)
        X_aug = tf.convert_to_tensor(X_aug)
        return X_red, X_aug

    def _concat_patch_samples(self, P0, P1):
        if len(P0) and len(P1):
            X_patch = tf.concat([P0, P1], axis=0)
        elif len(P0):
            X_patch = P0
        elif len(P1):
            X_patch = P1
        else:
            X_patch = tf.convert_to_tensor(())
        return X_patch

    def _data_augmentation(self, X, y, seg):
        grads, y_pred = get_batch_gradients(X, self.classifier, return_pred=True)
        tp_filter, fp_filter, fn_filter = self._filter_sample(X, y_pred, y)
        X_red, X_aug = self._generate_meaningful_samples(X, seg, grads, tp_filter,\
             red_flag=self.reduction, aug_flag=self.augmentation)
        if fp_filter is not None:
            P0, P1 = self._generate_meaningful_samples(X, seg, grads, fp_filter,\
                 red_flag=self.patch_reduction, aug_flag=self.patch_augmentation)
            X_patch = self._concat_patch_samples(P0, P1)
        else:
            X_patch = tf.convert_to_tensor(())
        return X_red, X_aug, X_patch, X[fn_filter]

    def _reduction_ce_loss(self, y_pred, agg):
        if agg == 'mean':
            bce = tf.keras.losses.BinaryCrossentropy()
        elif agg == 'sum': 
            bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        y_opp = tf.keras.utils.to_categorical([1-self.anomaly_class]*len(y_pred), num_classes=2)
        return bce(y_opp, y_pred)

    def _reduction_re_loss(self, y_pred, agg):
        red_prob = y_pred[:, self.anomaly_class]
        if agg == 'mean':
            return tf.reduce_mean(red_prob)
        elif agg == 'sum': 
            return tf.reduce_sum(red_prob)

    def _augmentation_ce_loss(self, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()
        y_ori = tf.keras.utils.to_categorical([self.anomaly_class]*len(y_pred), num_classes=2)
        return bce(y_ori, y_pred)

    def _augmentation_mse_loss(self, gradX, gradXa):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(gradX, gradXa)      

    def _reduction_loss(self, X_red, agg='mean'):
        y_r = self.classifier(X_red, training=True)
        if self.red_loss == 'CE':
            return self._reduction_ce_loss(y_r, agg)
        elif self.red_loss == 'RE':
            return self._reduction_re_loss(y_r, agg)
        else:
            return NotImplementedError

    def _augmentation_loss(self, X_aug, aug_loss=None):
        aug_loss = aug_loss if aug_loss is not None else self.aug_loss
        if aug_loss == 'CE':
            y_a = self.classifier(X_aug, training=True)
            return self._augmentation_ce_loss(y_a)
        elif aug_loss == 'MSE':
            # with tf.GradientTape() as g:
            #     g.watch(self.X1)
            #     y1 = self.classifier(self.X1)[:,self.anomaly_class]
            # gradX = tf.abs(g.gradient(y1, self.X1))
            # with tf.GradientTape() as g:
            #     g.watch(X_aug)
            #     y_a = self.classifier(X_aug)[:,self.anomaly_class]
            # gradXa = tf.abs(g.gradient(y_a, X_aug))    
            gradX = get_batch_gradients(self.X1, self.classifier)   
            gradXa = get_batch_gradients(X_aug, self.classifier)        
            return self._augmentation_mse_loss(gradX, gradXa)
        else:
            return NotImplementedError

    ################
    #   Override   #
    ################

    def call(self, x, training=None):
        return self.classifier(x, training=training)

    def train_step(self, data):
        X, y, s = data
        # X_red, X_aug = self._generate_meaningful_samples(X, y, s)
        X_red, X_aug, X_pat, X_fn = self._data_augmentation(X, y, s)
        
        with tf.GradientTape() as tape:
            y_0 = self.classifier(X, training=True)
            classifier_loss = self.compiled_loss(y, y_0, regularization_losses=self.losses)
            # fn_penalty = self._augmentation_loss(X_fn, aug_loss='CE') if len(X_fn) else tf.constant(0.)
            explainer_loss_red = self._reduction_loss(X_red) if len(X_red) else tf.constant(0.)
            explainer_loss_aug = self._augmentation_loss(X_aug) if len(X_aug) else tf.constant(0.)
            explainer_loss_pat = self._reduction_loss(X_pat, agg='sum') if len(X_pat) else tf.constant(0.)
            explainer_loss = self.reduction * explainer_loss_red + self.augmentation * explainer_loss_aug + self.patch * explainer_loss_pat

            loss = classifier_loss + explainer_loss #+ fn_penalty
        
        trainable_vars = self.trainable_variables
        gradTheta = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradTheta, trainable_vars))
        
        self.compiled_metrics.update_state(y, y_0)
        classifier_loss_tracker.update_state(classifier_loss)
        if self.reduction:
            red_loss_tracker.update_state(explainer_loss_red)
        if self.augmentation:
            aug_loss_tracker.update_state(explainer_loss_aug)
        if self.patch:
            pat_loss_tracker.update_state(explainer_loss_pat)
        explainer_loss_tracker.update_state(explainer_loss)
        total_loss_tracker.update_state(loss)
        
        metric_dict = {m.name: m.result() for m in self.metrics}
        if self.reg_trainable:
            reg_param_dict = {r.name: r.numpy() for r in self.regularization}
            metric_dict.update(reg_param_dict)
        return metric_dict

    def test_step(self, data):
        X, y, s = data
        X_red, X_aug, X_pat, X_fn = self._data_augmentation(X, y, s)
        y_0 = self.classifier(X)

        classifier_loss = self.compiled_loss(y, y_0, regularization_losses=self.losses)
        fn_penalty = self._augmentation_loss(X_fn, aug_loss='CE') if len(X_fn) else tf.constant(0.)
        explainer_loss_red = self._reduction_loss(X_red) if len(X_red) else tf.constant(0.)
        explainer_loss_aug = self._augmentation_loss(X_aug) if len(X_aug) else tf.constant(0.)
        explainer_loss_pat = self._reduction_loss(X_pat) if len(X_pat) else tf.constant(0.)
        explainer_loss = self.reduction * explainer_loss_red + self.augmentation * explainer_loss_aug + self.patch * explainer_loss_pat
        loss = classifier_loss + explainer_loss + fn_penalty

        self.compiled_metrics.update_state(y, y_0)
        classifier_loss_tracker.update_state(classifier_loss)
        if self.reduction:
            red_loss_tracker.update_state(explainer_loss_red)
        if self.augmentation:
            aug_loss_tracker.update_state(explainer_loss_aug)
        if self.patch:
            pat_loss_tracker.update_state(explainer_loss_pat)
        explainer_loss_tracker.update_state(explainer_loss)
        total_loss_tracker.update_state(loss)
        
        metric_dict = {m.name: m.result() for m in self.metrics}
        if self.reg_trainable:
            reg_param_dict = {r.name: r.numpy() for r in self.regularization}
            metric_dict.update(reg_param_dict)
        return metric_dict
        
    @property
    def regularization(self):
        _regs = []
        for r in [self.reduction, self.augmentation, self.patch]:
            if r:
                _regs.append(r)
        return _regs

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        _tracked = [total_loss_tracker, classifier_loss_tracker, explainer_loss_tracker]
        if self.reduction:
            _tracked.append(red_loss_tracker)
        if self.augmentation:
            _tracked.append(aug_loss_tracker)
        if self.patch:
            _tracked.append(pat_loss_tracker)
        return self.compiled_metrics.metrics + _tracked

    def get_config(self):
        config = {'frozen_layers': self.frozen_layers,
                 'anomaly_class': self.anomaly_class,
                 'mask_option': self.mask_option, 
                 'cutoff_option': self.cutoff_option, 
                 'cutoff_value': self.cutoff_value, 
                 'agg_option': self.agg_option,
                 'red_param': self.reduction.numpy(), 
                 'aug_param': self.augmentation.numpy(),
                 'pat_param': self.patch.numpy()
                 }
        if self.reduction:
            config.update({'red_loss': self.red_loss})
        if self.augmentation:
            config.update({'aug_loss': self.aug_loss})
        if self.patch:
            # patch_loss is the same with red_loss: the sample should remain benign after augmentation
            config.update({'pat_loss': self.red_loss,\
                 'patch_red': self.patch_reduction, 'patch_aug': self.patch_augmentation})
        return config


# def delete_model_embedding(m):
#     return tf.keras.Model(inputs=m.layers[1].input,outputs=m.outputs)
