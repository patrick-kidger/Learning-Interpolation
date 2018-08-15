"""Provides for testing and evaluating regressors."""

import numpy as np

# https://github.com/patrick-kidger/tools
import tools

from . import data_gen as dg


def _eval_regressor(regressor_factory, X, y):
    """Evaluates a regressor on some test data :X:, :y:.
    """
    
    regressor = regressor_factory()
    processor = regressor_factory.compile_kwargs.processor
    use_tf = regressor_factory.use_tf
    
    if use_tf:
        data_func = dg.BatchData.to_dataset((X, y))
    else:
        data_func = lambda: (X, y)
    
    with processor.training(False):
        predictor = regressor.predict(input_fn=data_func,
                                      yield_single_examples=False)
        prediction_before_postprocessing = next(predictor)
        prediction = processor.inverse_transform(X, prediction_before_postprocessing)
        
    diff = prediction - y
    squared_error = np.square(diff)
    total_loss = np.sum(squared_error)
    result = tools.Object(prediction=prediction,
                          X=X,
                          y=y,
                          diff=diff,
                          max_deviation=np.max(np.abs(diff)),
                          average_loss=np.mean(squared_error),
                          loss=total_loss / len(X),
                          total_loss=total_loss)
    return result

def _eval_regressors(regressor_factories, X, y):
    """Evaluates an iterable of regressors on some test data
    :X:, :y:."""
    results = []
    for regressor_factory in regressor_factories:
        result = _eval_regressor(regressor_factory, X, y)
        results.append(result)
        
    return results


def eval_regressor(regressor_factory, gen_one_data, batch_size=1):
    """Evaluates a regressor on some test data of size :batch_size:
    generated from :gen_one_data:.
    """
    X, y = dg.BatchData.batch(gen_one_data, batch_size)
    return _eval_regressor(regressor_factory, X, y)


def eval_regressors(regressor_factories, gen_one_data, batch_size=1):
    """Evaluates an iterable of regressors on some test data of size
    :batch_size: generated from :gen_one_data:.
    """
    X, y = dg.BatchData.batch(gen_one_data, batch_size)
    return _eval_regressors(regressor_factories, X, y)