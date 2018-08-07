"""Provides for creating DNN (specifically FFNN) from saved models."""

import functools as ft
import os

import tensorflow as tf
tflog = tf.logging

# https://github.com/patrick-kidger/tools
import tools

from . import factory as fac
from . import processor as pc


def _dnn_hyperparameters_from_dir(dir_name):
    """Creates DNN hyperparameters from the name of the directory of the DNN.
    """
    
    dnn_details = {}

    units, rest = dir_name.split('__')
    units = units.split('_')
    rest = rest.split('_')
    
    all_units = []
    for unit in units:
        if 'x' in unit:
            unit_size, unit_repeat = unit.split('x')
            unit_size, unit_repeat = int(unit_size), int(unit_repeat)
            all_units.extend([unit_size for _ in range(unit_repeat)])
        else:
            all_units.append(int(unit))
    dnn_details['hidden_units'] = all_units[:-1]
    dnn_details['logits'] = all_units[-1]
    
    processor_name = rest[0]
    processor_class = pc.ProcessorBase.find_subclass(processor_name)
    dnn_details['processor'] = processor_class()
    dnn_details['batch_norm'] = False
    
    activation_name = rest[1].lower()
    
    # Not a great way to do this inversion, admittedly
    if activation_name[:9] == 'leakyrelu':
        alpha = float(str(activation_name[9]) + '.' + str(activation_name[10:]))
        dnn_details['activation'] = ft.partial(tf.nn.leaky_relu, alpha=alpha)
    else:
        try:
            activation_fn = getattr(tf.nn, activation_name)
        except AttributeError:
            raise RuntimeError("Activation '{}' not understood.".format(activation_name))
        else:
            dnn_details['activation'] = activation_fn
        
    remaining = rest[2:]
    if len(remaining) == 0:
        uuid = None
    elif len(remaining) == 1:
        uuid = remaining[0]
    else:
        raise RuntimeError("Bad dir_name string '{}'. Too many remaining "
                           "arguments: {}".format(dir_name, remaining))
        
    return dnn_details, uuid


def dnn_factory_from_model_dir(model_dir, **kwargs):
    """Creates a DNN from the :model_dir: argument. Any additional keyword
    arguments provided override the details of the DNN found."""
    
    if model_dir[-1] in ('/', '\\'):
        model_dir = model_dir[:-1]
    model_dir_split = tools.split(['/', '\\'], model_dir)
    dir_name = model_dir_split[-1]
    # I suspect that we should be able to restore the DNN just from the
    # information saved in the model directory, without needing to know
    # its structure from the directory name...
    dnn_details, uuid = _dnn_hyperparameters_from_dir(dir_name)
    dnn_details.update(kwargs)
    dnn_factory = fac.DNNFactory(model_dir=model_dir, **dnn_details)
    return dnn_factory


def dnn_factories_from_dir(dir_, exclude_start=('.',), exclude_end=(), 
                           exclude_in=(), **kwargs):
    """Creates multiple DNNs and processors from a directory containing the
    directories for multiple DNNs and processors.
    
    Its arguments :exclude_start:, :exclude_end:, :exclude_in: are each
    tuples which allow for excluding particular models, if their model 
    directories start, end, or include any of the strings specified
    in each tuple respectively.
    
    Essentially just a wrapper around dnn_factory_from_model_dir, to run it
    multiple times. It will forward any additional keyword arguments onto
    each call of dnn_factory_from_model_dir.
    """
    
    subdirectories = sorted(next(os.walk(dir_))[1])
    if dir_[-1] in ('/', '\\'):
        dir_ = dir_[:-1]
    dnn_factories = []
    names = []
    
    for subdir in subdirectories:
        if any(subdir.startswith(ex) for ex in exclude_start):
            tflog.info("Excluding '{}' based on start.".format(subdir))
            continue
        if any(subdir.endswith(ex) for ex in exclude_end):
            tflog.info("Excluding '{}' based on end.".format(subdir))
            continue
        if any(ex in subdir for ex in exclude_in):
            tflog.info("Excluding '{}' based on containment.".format(subdir))
            continue
            
        model_dir = dir_ + '/' + subdir
        try:
            dnn_factory = dnn_factory_from_model_dir(model_dir, **kwargs)
        except (FileNotFoundError, RuntimeError) as e:
            tflog.info("Could not load DNN from '{}'. Error message: '{}'"
                       .format(subdir, e))
        else:
            dnn_factories.append(dnn_factory)
            names.append(subdir)
            
    return dnn_factories, names
