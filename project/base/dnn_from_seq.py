"""Provides for creating DNNs (specifically FFNN) ab initio."""

import functools as ft
import itertools as it

import tensorflow as tf
tfe = tf.estimator
tflo = tf.losses
tft = tf.train

from . import processor as pc


# Keras-inspired nice interface, just without the slow speed and lack of 
# multicore functionality of Keras...
# (Plus it allows us to integrate our preprocessing)
class Sequential:
    """Defines a neural network. Expected usage is roughly:
    
    >>> model = Sequential()
    >>> model.add(tf.layers.Dense(units=100, activation=tf.nn.relu))
    >>> model.add_train(tf.layers.Dropout(rate=0.4))
    >>> model.add(tf.layers.Dense(units=50, activation=tf.nn.relu))
    >>> model.add_train(tf.layers.Dropout(rate=0.4))
    >>> model.add(tf.layers.Dense(units=10, activation=tf.nn.relu))
    
    to define the neural network in the abstract (note that the last dense layer
    are treated as the logits), followed by:
    
    >>> dnn = model.compile()
    
    to actually create it in TensorFlow. Here, 'dnn' is a tf.Estimator, so may
    be used like:
    
    >>> dnn.train(...)
    >>> dnn.predict(...)
    >>> dnn.evaluate(...)
    """
    
    def __init__(self):
        """Creates a Sequential. See Sequential.__doc__ for more info."""
        self._layer_funcs = []
        self._layer_train = []
        
    def add(self, layer):
        """Add a layer to the network.
        """
        self._layer_funcs.append(layer)
        self._layer_train.append(False)
        
    def add_train(self, layer):
        """Add a layer to the network which needs to know if the network is in
        training or not.
        """
        self.add(layer)
        self._layer_train[-1] = True
        
    def compile(self, optimizer=None, loss_fn=tflo.mean_squared_error, 
                model_dir=None, gradient_clip=None, processor=None, **kwargs):
        """Takes its abstract neural network definition and compiles it into a
        tf.estimator.Estimator.
        
        May be given an :optimizer:, defaulting to tf.train.AdamOptimizer().
        May be given a :loss_fn:, defaulting to tf.losses.mean_squared_error.
        May be given a :gradient_clip:, defaulting to no clipping.
        May be given a :processor:, which will be saved and loaded.
        
        Any additional kwargs are passed into the creation of the
        tf.estimator.Estimator.
        """
        
        # Probably shouldn't use the same optimizer instance every time? Hence
        # this.
        if optimizer is None:
            optimizer = tft.AdamOptimizer()
            
        if processor is None:
            processor = pc.IdentityProcessor()
            
        def model_fn(features, labels, mode):
            # Create processor variables
            processor.init(model_dir)
            
            # Apply any preprocessing to the features and labels
            features, labels = processor.transform(features, labels)
            
            # First layer is the feature inputs.
            layers = [features]
            
            for prev_layer, layer_func, train in zip(layers, self._layer_funcs, 
                                                     self._layer_train):
                if train:
                    layer = layer_func(inputs=prev_layer, 
                                       training=mode == tfe.ModeKeys.TRAIN)
                else:
                    layer = layer_func(inputs=prev_layer)
                    
                # Deliberately using the generator nature of zip to add elements
                # to the layers list as we're iterating through it.
                # https://media.giphy.com/media/3oz8xtBx06mcZWoNJm/giphy.gif
                layers.append(layer)
                
            logits = layers[-1]
            
            if mode == tfe.ModeKeys.PREDICT:
                return tfe.EstimatorSpec(mode=mode, predictions=logits)
            
            loss = loss_fn(labels, logits)

            if mode == tfe.ModeKeys.TRAIN:
                g_step = tft.get_global_step()
                if gradient_clip is None:
                    train_op = optimizer.minimize(loss=loss, global_step=g_step)
                else:
                    # Perform Gradient clipping
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        gradients, variables = zip(*optimizer.compute_gradients(loss))
#                         gradients0 = tf.Print(gradients[0], [tf.global_norm(gradients)], 'Global norm: ')
#                         gradients = tuple([gradients0, *gradients[1:]])
                        gradients, _ = tf.clip_by_global_norm(gradients, 
                                                              gradient_clip)
                        train_op = optimizer.apply_gradients(zip(gradients, 
                                                                 variables),
                                                             global_step=g_step)
                training_hooks = [] if model_dir is None else [pc.ProcessorSavingHook(processor, model_dir)]
                return tfe.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                         training_hooks=training_hooks)
            
            if mode == tfe.ModeKeys.EVAL:
                return tfe.EstimatorSpec(mode=mode, loss=loss)
            
            raise ValueError("mode '{}' not understood".format(mode))
                
        return tfe.Estimator(model_fn=model_fn, model_dir=model_dir, **kwargs)
    
    
def model_dir_str(model_dir, hidden_units, logits, processor, activation, 
                  uuid=None):
    """Returns a string for the model directory describing the network.
    
    Note that it only stores the information that describes the layout of
    the network - in particular it does not describe any training
    hyperparameters (in particular dropout rate).
    """
    
    layer_counter = [(k, sum(1 for _ in g)) for k, g in it.groupby(hidden_units)]
    for layer_size, layer_repeat in layer_counter:
        if layer_repeat == 1:
            model_dir += '{}_'.format(layer_size)
        else:
            model_dir += '{}x{}_'.format(layer_size, layer_repeat)
    model_dir += '{}__'.format(logits)
    model_dir += processor.__class__.__name__
    
    if isinstance(activation, ft.partial):
        activation_fn = activation.func
        alpha = str(activation.keywords['alpha']).replace('.', '')
    else:
        activation_fn = activation
        alpha = '02'
        
    model_dir += '_' + activation_fn.__name__.replace('_', '')
    if activation_fn is tf.nn.leaky_relu:
        model_dir += alpha

    if uuid not in (None, ''):
        model_dir += '_' + str(uuid)
    return model_dir
