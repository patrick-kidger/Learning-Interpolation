"""Provides simple interpolation methods; useful to give a baseline to compare
neural network models against."""

import numpy as np
import sklearn.base as skb
import sklearn.preprocessing as skpr
import sklearn.pipeline as skpi
import sklearn.linear_model as sklm

# https://github.com/patrick-kidger/tools
import tools

from . import grid


class InterpolatorBase:
    """Base class for performing predictions based on just the input. Subclasses
    are expected to provide a predict_single classmethod specifying their
    predictions.
    
    Its predict and evaluate methods are designed to resemble that of
    tf.estimator.Estimator's, so that we can call them in the same way. (We don't
    actually inherit from tf.estimator.Estimator because none of what these 
    classes use TensorFlow, so messing around with model functions and 
    EstimatorSpecs is just unnecessary faff and overhead.)
    
    WARNING: All subclasses must expect no preprocessing, i.e. must use
    IdentityProcessor() as their preprocessing. This is because preprocessing is
    done in TensorFlow, which, of course, this class is explicitly about not 
    using... if any preprocessing is necessary then subclasses must implement it
    themselves as part of their predict_single method.
    """
        
    def _prepare(self, Xi):
        """Performs any necessary preparations on the data :Xi: before making 
        predictions.
        """
        pass
    
    def _interp(self, Xi, point):
        """Helper function for performing interpolation on a coarse
        grid :Xi:, giving the value of the interpolation at :point:.
        
        The spacing of the grid is known from the global hyperparameters
        defining the coarse grid size, whilst it isn't necessary to know its
        location.
        
        The argument :point: should be scaled to the grid size, i.e.
        coarse_grid_sep.
        """
        raise NotImplementedError
    
    def predict_single(self, Xi, y):
        """Makes a prediction corresponding to input feature :Xi:.
        
        It is given the true result :y:. Not to cheat and return perfect
        results, but to determine its shape etc.
        """
        raise NotImplementedError
    
    def predict(self, input_fn, yield_single_examples=False):
        """The argument :input_fn: should probably be a lambda wrapper around
        the result of BatchData.batch.
        
        The argument :yield_single_examples: is there for compatibility with the
        interface for the usual TF Estimators and is ignored.
        """
        
        returnval = []
        X, y = input_fn()
        
        for Xi, yi in zip(X, y):
            returnval.append(self.predict_single(Xi, yi))
            
        returnval = np.array(returnval)
        while True:
            yield returnval

            
class InterpolatorGrid(InterpolatorBase):
    """Provides the predict_single function for predictions on a fine grid.
    
    Requires the _prepare and _interp methods provided by one of the mixins
    above."""
    
    def predict_single(self, Xi, yi):
        returnval = []
        # Translation doesn't matter at this point so WLOG the fine grid is
        # around 0, 0. (cls._interp makes the same assumption; these assumptions
        # must be consistent)
        for point in grid.fine_grid((0, 0)):
            self._prepare(Xi)
            returnval.append(self._interp(Xi, point))
        return returnval
    
    
class InterpolatorPoint(InterpolatorBase):
    """Provides the predict_single function for predictions at a single point.
    
    Requires the _prepare and _interp methods provided by one of the mixins
    above."""
    
    def predict_single(self, Xi, yi):
        # Separate the location data and the grid data
        t_offset = Xi[-2]
        x_offset = Xi[-1]
        Xi = Xi[:-2]
        self._prepare(Xi)
        # Wrapped in a list for consistency: this network just happens to only
        # be trying to predict a single label.
        return [self._interp(Xi, (t_offset * grid.coarse_grid_sep.t, 
                                  x_offset * grid.coarse_grid_sep.x))]
    
    
class BilinearInterpMixin(InterpolatorBase):
    """Mixin to help perform bilinear interpolation."""
        
    def _interp(self, Xi, point):        
        # The actual t, x values for the grid don't matter from this point 
        # onwards; so this is just a translation from wherever X was actually 
        # calculated. So WLOG assume it was around 0.
        cg = grid.coarse_grid((0, 0))
        t, x = point
        
        # The grid points nearest :point:.
        t_below = tools.round_mult(t, grid.coarse_grid_sep.t, 'down')
        t_above = tools.round_mult(t, grid.coarse_grid_sep.t, 'up')
        x_below = tools.round_mult(x, grid.coarse_grid_sep.x, 'down')
        x_above = tools.round_mult(x, grid.coarse_grid_sep.x, 'up')
        
        # The value of :Xi: at those grid points.
        t_b_x_b = Xi[grid._index_tol(cg, (t_below, x_below))]
        t_a_x_b = Xi[grid._index_tol(cg, (t_above, x_below))]
        t_b_x_a = Xi[grid._index_tol(cg, (t_below, x_above))]
        t_a_x_a = Xi[grid._index_tol(cg, (t_above, x_above))]
        
        # Shift the t, x values to be relative to the bottom-left point of the
        # grid square in which (t, x) lies.
        t_scale = (t % grid.coarse_grid_sep.t) / grid.coarse_grid_sep.t
        x_scale = (x % grid.coarse_grid_sep.x) / grid.coarse_grid_sep.x
        
        # Bilinear interpolation
        returnval = (1 - t_scale) * (1 - x_scale) * t_b_x_b
        returnval += t_scale * (1 - x_scale) * t_a_x_b
        returnval += (1 - t_scale) * x_scale * t_b_x_a
        returnval += t_scale * x_scale * t_a_x_a
        
        return returnval
    
    
def poly(point, poly_coefs, poly_deg):
    """Interprets the given :poly_coefs: as a polynomial, and 
    evaluates them at the specified :point:.
    The argument :poly_deg: is used to check that poly_coefs
    is of a suitable length to be interpreted as polynomial
    coefficients.
    """

    t, x = point
    coefs = iter(poly_coefs)

    result = next(coefs)  # Intercept, i.e. constant term
    for power in range(1, poly_deg + 1):
        for x_power in range(0, power + 1):
            t_power = power - x_power
            coef = next(coefs)
            result += coef * (t ** t_power) * (x ** x_power)
    try:
        next_coef = next(coefs)
    except StopIteration:
        return result
    else:
        raise RuntimeError('coef_: {coef_}, poly_deg: {poly_deg}, '
                           'coef that shouldn\'t exist: {next_coef}'
                           .format(coef_=coef_, 
                                   poly_deg=poly_deg, 
                                   next_coef=next_coef))
    
    
class PolyInterpMixin(InterpolatorBase):
    """Mixin to help perform polynomial interpolation."""
    
    def __init__(self, poly_deg, *args, **kwargs):
        self.poly_deg = poly_deg
        self._poly_coefs = None
        super(PolyInterpMixin, self).__init__(*args, **kwargs)
        
    def poly(self, point):
        """Interprets its currently stored polynomial coefficients as a 
        polynomial, and evaluates them at the specified :point:.
        """
        
        if self._poly_coefs is None:
            raise RuntimeError('Must run _prepare first!')
        # poly is defined separately that other things can use it.
        return poly(point, self._poly_coefs, self.poly_deg)
    
    def _prepare(self, Xi):
        poly_features = skpr.PolynomialFeatures(degree=self.poly_deg, 
                                                include_bias=True)
        lin_reg = sklm.LinearRegression(fit_intercept=False)
        poly_pipe = skpi.Pipeline([('pf', poly_features), ('lr', lin_reg)])
        
        # The actual t, x values for the grid don't matter from this point 
        # onwards; so this is just a translation from wherever X was actually 
        # calculated. So WLOG assume it was around 0.
        cg = grid.coarse_grid((0, 0))
        poly_pipe.fit(cg, Xi)
        self._poly_coefs = poly_pipe.named_steps['lr'].coef_
        
    
    def _interp(self, Xi, point):
        return self.poly(point)
    
    
class NearestInterpMixin(InterpolatorBase):
    """Mixin to help perform nearest-neighbour interpolation."""
        
    def _interp(self, Xi, point):        
        # The actual t, x values for the grid don't matter from this point 
        # onwards; so this is just a translation from wherever X was actually 
        # calculated. So WLOG assume it was around 0.
        cg = grid.coarse_grid((0, 0))
        t, x = point
        
        # The grid point nearest :point:.
        t_nearest = tools.round_mult(t, grid.coarse_grid_sep.t, 'round')
        x_nearest = tools.round_mult(x, grid.coarse_grid_sep.x, 'round')
        
        # The value of :Xi: at those grid points.
        t_n_x_n = Xi[grid._index_tol(cg, (t_nearest, x_nearest))]

        return t_n_x_n
    
    
class BilinearInterpGrid(BilinearInterpMixin, InterpolatorGrid):
    pass


class PolyInterpGrid(PolyInterpMixin, InterpolatorGrid):
    pass


class BilinearInterpPoint(BilinearInterpMixin, InterpolatorPoint):
    pass


class PolyInterpPoint(PolyInterpMixin, InterpolatorPoint):
    pass
    
    
class Perfect(InterpolatorBase):
    """Regressor that cheats to always give the perfect prediction."""
    
    def predict_single(self, Xi, yi):
        return yi
