"""Provides a regressor averager to handle an ensemble of regressors."""

import numpy as np

import tensorflow as tf
tflog = tf.logging

# https://github.com/patrick-kidger/tools
import tools

from . import dnn_from_folder as d_f
from . import evalu


class RegressorAverager:
    """Regressor that averages the results of other regressors to make its
    prediction. It is capable of using both TensorFlow-based regressors
    and non-TensorFlow-based regressors as it expects regressor factories,
    and the difference between them is handled by the factory wrapper.
    """
    
    def __init__(self, regressor_factories, mask=None, **kwargs):
        """Should be passed an iterable of :regressor_factories:. It will
        make predictions according to their average.
        
        May also pass a :mask: argument, which should be a tuple of bools
        the same length as the number of regressors, specifying whether or
        not particular regressor should be used when making predictions.
        """
        self.regressor_factories = tuple(regressor_factories)
        self.mask = None
        self.reset_mask()
        if mask is not None:
            self.set_mask(mask)
        super(RegressorAverager, self).__init__(**kwargs)
        
    def set_mask(self, mask):
        """Sets a mask to only use some of the regressors."""
        assert len(mask) == len(self.regressor_factories)
        if not mask:
            tflog.warn('Setting empty mask for {}.'.format(self.__class__.__name__))
        self.mask = tuple(mask)
        return self  # for chaining
        
    def reset_mask(self):
        """Resets the mask, so as to use all regressors."""
        self.mask = [True for _ in range(len(self.regressor_factories))]
        return self  # for chaining
    
    def auto_mask(self, gen_one_data, batch_size, thresh=None, top=None):
        """Automatically creates a mask to only use the regressors that
        are deemed to be 'good'.
        
        The function :gen_one_data: will be called :batch_size: times to 
        generate the  (feature, label) pairs on which the regressors are 
        tested.
        
        At least one of :thresh: or :top: should be passed. The best :top:
        number of regressors whose loss is smaller than :thresh: will be
        deemed to be 'good'. If :thresh: is None (which it defaults to),
        then simply the best :top: regressors will be used. If :top: is
        None (which it defaults to) then simply every regressor whose loss
        is at least :thresh: will be used.
        """
        if thresh is None and top is None:
            raise RuntimeError('At least one of thresh or top must be not '
                               'None.')
            
        if thresh is None:
            thresh = np.inf
        if top is None:
            top = len(self.regressor_factories)
        top = top - 1
        
        results = evalu.eval_regressors(self.regressor_factories, 
                                        gen_one_data, batch_size)
        loss_values = [result.loss for result in results]            
        thresh_loss = min(sorted(loss_values)[top], thresh)
        
        dnn_mask = []
        for loss in loss_values:
            dnn_mask.append(loss <= thresh_loss)
            
        self.set_mask(dnn_mask)
        
    def predict(self, input_fn, *args, **kwargs):
        X, y = input_fn()
        
        returnval = tools.AddBase()
        counter = 0
        for regressor_factory, mask in zip(self.regressor_factories, self.mask):
            if mask:
                counter += 1
                returnval += evalu._eval_regressor(regressor_factory, X, y).prediction
        returnval = returnval / counter
        
        while True:
            yield returnval

    @classmethod
    def from_dir(cls, dir_, gen_data=None, thresh=None, batch_size=1000):
        """Creates an average regressor from all regressors in a specified
        directory.
        """
        
        dnn_factories, names = d_f.dnn_factories_from_dir(dir_)
        self = cls(regressor_factories=dnn_factories)
        if gen_data is not None and thresh is not None:
            self.auto_mask(gen_data, thresh)
            
        return self
            