"""Processors handle data preprocessing before it's fed into the network."""

import json
import numpy as np

import tensorflow as tf
tflog = tf.logging
tft = tf.train

# https://github.com/patrick-kidger/tools
import tools

from . import grid


class ProcessorBase(tools.SubclassTrackerMixin('__name__')):
    """Base class for preprocessors."""
    
    save_attr = []
    checkpoint_filename = 'processor-checkpoint.ckpt'
    
    def __init__(self, training=True, **kwargs):
        self._training = training
        super(ProcessorBase, self).__init__(**kwargs)
        
    def init(self, model_dir):
        """Initialises Python and TensorFlow variables."""
        # super().init() should be called before the subclass' init()
        self.load(model_dir)
        
    def training(self, val):
        """Provides a context to set the training variable to :val:."""
        return tools.set_context_variables(self, ('_training',), val)
    
    def transform(self, X, y):
        """Processes the data."""
        # Note that y may be None during prediction; make sure transform is
        # appropriately defined.
        raise NotImplementedError
        
    def inverse_transform(self, X, y):
        """Performs the inverse transform on the data."""
        raise NotImplementedError
    
    def save(self, session, step, model_dir):
        """Saves the processor to a file in the directory :model_dir:. The argument
        :step: is logged out to specify at what global step this was performed."""
        
        if self.save_attr:
            file_loc = model_dir + '/' + self.checkpoint_filename

            # Sync the Python version of the variables to the TensorFlow version, so
            # that the variables take their correct values if inverse_transform is
            # called next (e.g. if we've just finished training and are now predicting)
            for name in self.save_attr:
                tf_variable = getattr(self, name + '_tf')
                value = tf_variable.eval(session=session)
                setattr(self, name, value)

            # And now also use those eval()'d variables to save their values.
            if model_dir is not None:
                def val(name):
                    returnval = getattr(self, name)
                    # Can't convert np.bool_ to JSON.
                    if isinstance(returnval, np.bool_):
                        return bool(returnval)
                    else:
                        return returnval
                write_dict = {name: val(name) for name in self.save_attr}
                with open(file_loc, 'w') as f:
                    f.write(json.dumps(write_dict))

            tflog.info('Saving processor checkpoint for {} into {}'.format(step, file_loc))
        
    def load(self, model_dir):
        """Sets the processor's variables to what is specified in the save file
        located in the directory :model_dir:.
        """
        
        if self.save_attr:
            if model_dir is None:
                tflog.warn("Passed None as model_dir and so cannot load file.")
            else:
                file_loc = model_dir + '/' + self.checkpoint_filename

                try:
                    with open(file_loc, 'r') as f:
                        load_dict = json.loads(f.read())
                except FileNotFoundError:
                    tflog.info("No processor checkpoint file {} found.".format(file_loc))
                except json.JSONDecodeError:
                    tflog.warn("Ignoring JSONDecodeError on attempting to load file {}."
                               .format(file_loc))
                else:
                    tflog.info("Restoring processor parameters from {}".format(file_loc))
                    for key, val in load_dict.items():
                        setattr(self, key, val)
        
    
class IdentityProcessor(ProcessorBase):
    """Performs no processing."""
    
    def transform(self, X, y):
        return X, y
    
    def inverse_transform(self, X, y):
        return y
  
    
class ScaleOverall(ProcessorBase):
    """Scales data to between -1 and 1. Scaling is done across all batches."""
    
    save_attr = ['mean', 'extent', 'momentum', '_started']
    
    def __init__(self, momentum=0.99, **kwargs):
        self.momentum = momentum
        self.mean = 0.0
        self.extent = 1.0
        self._started = False
        super(ScaleDataOverall, self).__init__(**kwargs)
        
    def init(self, model_dir):
        super(NormalisationOverall, self).init(model_dir)
        self.momentum_tf = tf.Variable(self.momentum, trainable=False, dtype=tf.float64)
        self.mean_tf = tf.Variable(self.mean, trainable=False, dtype=tf.float64)
        self.extent_tf = tf.Variable(self.extent, trainable=False, dtype=tf.float64)
        self._started_tf = tf.Variable(self._started, trainable=False)
        
    def transform(self, X, y):
        def first_time():
            assignment = self._started_tf.assign(True)
            with tf.control_dependencies([assignment]):
                mean = tf.reduce_mean(X)
            extent = tf.reduce_max(tf.abs(X - mean))
            m = self.mean_tf.assign(mean)
            s = self.extent_tf.assign(extent)
            return m, s
        
        def later_times():
            mean = tf.reduce_mean(X)
            extent = tf.reduce_max(tf.abs(X - mean))
            m = self.mean_tf.assign(self.mean_tf * self.momentum_tf + mean * (1 - self.momentum_tf))
            s = self.extent_tf.assign(self.extent_tf * self.momentum_tf + extent * (1 - self.momentum_tf))
            return m, s
        
        if self._training:
            mean, extent = tf.cond(tf.equal(self._started_tf, False), first_time, later_times)
        else:
            mean, extent = self.mean_tf, self.extent_tf
        
        X_scaled = (X - mean) / extent
        y_scaled =  None if y is None else (y - mean) / extent
        return X_scaled, y_scaled
    
    def inverse_transform(self, X, y):
        return (y * self.extent) + self.mean
    
    
class NormalisationOverall(ProcessorBase):
    """Normalises inputs by subtracting mean and dividing by standard deviation.
    Scaling is done across all batches.
    """
    
    save_attr = ['mean', 'stddev', 'momentum', '_started']
    
    def __init__(self, momentum=0.99, **kwargs):
        self.momentum = momentum
        self.mean = 0.0
        self.stddev = 1.0
        self._started = False
        super(NormalisationOverall, self).__init__(**kwargs)
        
    def init(self, model_dir):
        super(NormalisationOverall, self).init(model_dir)
        self.momentum_tf = tf.Variable(self.momentum, trainable=False, dtype=tf.float64)
        self.mean_tf = tf.Variable(self.mean, trainable=False, dtype=tf.float64)
        self.stddev_tf = tf.Variable(self.stddev, trainable=False, dtype=tf.float64)
        self._started_tf = tf.Variable(self._started, trainable=False)
        
    def transform(self, X, y):
        def first_time():
            assignment = self._started_tf.assign(True)
            with tf.control_dependencies([assignment]):
                mean = tf.reduce_mean(X)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(X - mean)))
            m = self.mean_tf.assign(mean)
            s = self.stddev_tf.assign(stddev)
            return m, s
        
        def later_times():
            mean = tf.reduce_mean(X)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(X - mean)))
            m = self.mean_tf.assign(self.mean_tf * self.momentum_tf + mean * (1 - self.momentum_tf))
            s = self.stddev_tf.assign(self.stddev_tf * self.momentum_tf + stddev * (1 - self.momentum_tf))
            return m, s
        
        if self._training:
            mean, stddev = tf.cond(tf.equal(self._started_tf, False), first_time, later_times)
        else:
            mean, stddev = self.mean_tf, self.stddev_tf
            
        X_scaled = (X - mean) / stddev
        y_scaled = None if y is None else (y - mean) / stddev
        return X_scaled, y_scaled
    
    def inverse_transform(self, X, y):
        return (y * self.stddev) + self.mean
    
    
class ScaleCenter(ProcessorBase):
    """Scales input so that everything is scaled so that the
    four grid points at the corners of the fine grid have
    minimum 0 and maximum 1.
    """
    
    def _transform(self, Xi_yi):
        Xi, yi = Xi_yi
        max_ = Xi[grid.coarse_grid_center_indices[0]]
        min_ = Xi[grid.coarse_grid_center_indices[0]]
        for i in grid.coarse_grid_center_indices[1:]:
            max_ = tf.maximum(Xi[i], max_)
            min_ = tf.minimum(Xi[i], min_)
        extent = max_ - min_
        Xi_scaled = (Xi - min_) / extent
        yi_scaled = (yi - min_) / extent
        return Xi_scaled, yi_scaled
    
    def transform(self, X, y):        
        X_scaled, y_scaled = tf.map_fn(self._transform, (X, y), back_prop=False)
        return X_scaled, y_scaled
    
    def inverse_transform(self, X, y):
        y_scaled = []
        for Xi, yi in zip(X, y):
            max_ = Xi[grid.coarse_grid_center_indices[0]]
            min_ = Xi[grid.coarse_grid_center_indices[0]]
            for i in grid.coarse_grid_center_indices[1:]:
                max_ = max(Xi[i], max_)
                min_ = min(Xi[i], min_)
            extent = max_ - min_
            y_scaled.append(yi * extent + min_)
        return np.array(y_scaled)
    
    
class AtPointMixin(ProcessorBase):
    """Adapts the transform method to only act on the [:-2] elements 
    of X; i.e. omitting the extra (t, x) pair that is given to it.
    """
    def transform(self, X, y):
        X_len = tf.shape(X)[1]
        # axis 0 is the batch size
        # axis 1 is the number of features
        X_, t, x = tf.split(X, [X_len - 2, 1, 1], axis=1)
        X_, y = super(AtPointMixin, self).transform(X_, y)
        X = tf.concat([X_, t, x], axis=1)
        # Complete hack.
        # The whole way of handling the extra (t, x) data needs redoing
        # to use dictionaries.
        num_features = (grid.num_intervals.t + 1) * (grid.num_intervals.x + 1) + 2
        X.set_shape((None, num_features))
        return X, y
    
    
class ScaleCenterAtPoint(AtPointMixin, ScaleCenter):
    pass
                                            
    
    
### Hooks

class ProcessorSavingHook(tft.SessionRunHook):
    """Saves the processor data."""
    # Adapted from the source code for tf.train.CheckpointSaverHook
    
    def __init__(self, processor, model_dir, save_secs=600, 
                 save_steps=None, **kwargs):
        self.processor = processor
        self.model_dir = model_dir
        self._timer = tft.SecondOrStepTimer(every_secs=save_secs,
                                            every_steps=save_steps)
        self._global_step_tensor = None
        super(ProcessorSavingHook, self).__init__(**kwargs)
    
    def begin(self):
        self._global_step_tensor = tft.get_global_step()
        
    def after_create_session(self, session, coord):
        global_step = session.run(self._global_step_tensor)
        self._save(session, global_step)
        self._timer.update_last_triggered_step(global_step)
        
    def before_run(self, run_context):
        return tft.SessionRunArgs(self._global_step_tensor)
        
    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                self._save(run_context.session, global_step)
            
    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            self._save(session, last_step)
        
    def _save(self, session, step):
        self.processor.save(session, step, self.model_dir)
