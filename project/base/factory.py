"""TensorFlow demands that its Estimators be rebuilt before each train(...)
etc. call, so DNNFactory is necessary to construct it each time. For a 
consistent interface, RegressorFactory is also provided to wrap around 
non-TensorFlow regressors.
"""

import tensorflow as tf
tfe = tf.estimator
tfi = tf.initializers
tfla = tf.layers

# https://github.com/patrick-kidger/tools
import tools

from . import dnn_from_seq as d_s
from . import processor as pc


class RegressorFactoryBase:
    """Defines the interface for factories which make regressors, i.e. DNNs
    or simple interpolators.
    """
    # No good way to set abstract instance attributes (even with @property)
    # so we just list them in comments here. (Best way would probably be to
    # check for the attributes existence in the metaclass' __call__, which
    # is easily more faff than it's worth.)
    
    # Instances should have a 'processor' attribute returning the processor
    # for preprocessing input data to the regressor
    
    # Instances should have a 'use_tf' attribute returning True or False
    # for whether the regressor uses TensorFlow
    
    def __call__(self, **kwargs):
        # Should return the regressor itself
        raise NotImplementedError
        
            
class DNNFactory(RegressorFactoryBase):
    """Shortcut for creating a simple DNN with dense, dropout and batch 
    normalization layers, and then compiling it.
    
    The reason its __call__ function has this class wrapper is because of
    how TensorFlow operates: the DNN needs to be recreated before every
    train(...), predict(...) or evaluate(...) call, so it is convenient to
    cache the hyperparameters for the DNN in this class, and simply call
    the class before each train/predict/evaluate call.
    """
    
    def __init__(self, hidden_units, logits, processor=None, 
                 activation=tf.nn.relu, drop_rate=0.0, 
                 drop_type='dropout', log_steps=100,
                 batch_norm=False, 
                 kernel_initializer=tfi.truncated_normal(mean=0, stddev=0.05),
                 compile_kwargs=None,
                 **kwargs):
        if compile_kwargs is None:
            compile_kwargs = {}
        self.hidden_units = hidden_units
        self.logits = logits
        self.activation = activation
        self.drop_rate = drop_rate
        self.drop_type = drop_type
        self.log_steps = log_steps
        self.batch_norm = batch_norm
        self.kernel_initializer = kernel_initializer
        self.compile_kwargs = tools.Object.from_dict(compile_kwargs)
        self.use_tf = True
        super(DNNFactory, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        model = d_s.Sequential()
        if self.batch_norm:
            model.add_train(tfla.BatchNormalization())
        for units in self.hidden_units:
            model.add(tfla.Dense(units=units, activation=self.activation,
                                 kernel_initializer=self.kernel_initializer))
            if self.batch_norm:
                model.add_train(tfla.BatchNormalization())
            if self.drop_rate != 0:
                if self.drop_type in ('normal', 'dropout'):
                    model.add_train(tfla.Dropout(rate=self.drop_rate))
                elif self.drop_type in ('alpha', 'alpha_dropout'):
                    model.add_train(tfla.AlphaDropout(rate=self.drop_rate))
        model.add(tfla.Dense(units=self.logits, 
                             kernel_initializer=self.kernel_initializer))
        
        tools.update_without_overwrite(kwargs, self.compile_kwargs)
        return model.compile(config=tfe.RunConfig(log_step_count_steps=self.log_steps),
                             **kwargs)
    
    
class RegressorFactory(RegressorFactoryBase):
    """Factory wrapper around any regressor which doesn't use TensorFlow; i.e.
    the interpolators above, or RegressorAverager (whether or not
    RegressorAverager itself is using TensorFlow-based regressors.)"""
    
    def __init__(self, regressor, **kwargs):
        self.regressor = regressor
        self.processor = pc.IdentityProcessor()
        self.use_tf = False
        super(RegressorFactory, self).__init__(**kwargs)
        
    def __call__(self, **kwargs):
        return self.regressor
