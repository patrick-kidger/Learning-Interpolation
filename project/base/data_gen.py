"""Provides for generating and batching data, and converting it into the
tf.data.Dataset format expected by TensorFlow.
"""

import functools as ft
import multiprocessing as mp
import numpy as np

import tensorflow as tf
tfd = tf.data
tflog = tf.logging

# https://github.com/patrick-kidger/tools
import tools

from . import grid


### Solutions to the Camassa--Holm equation

class SolutionBase:
    def __call__(self, point):
        """Evaluates the solution at the particular point."""
        raise NotImplementedError
        
    def on_grid(self, grid, extra=0):
        """Evaluates the solution on a grid of points.
        
        :[(float, float)] grid: The grid points on which to evaluate, as
            returned by either fine_grid or coarse_grid.
        :int extra: A nonnegative integer specifying that the array should be
            larger by this many entries. (To add extra data to later without the
            overheard of creating another array.)"""
        return np.array([self(grid_point) for grid_point in grid] 
                        + [0 for _ in range(extra)])
    
    def on_vals(self, tvals, xvals):
        """Evalutes the solution on a grid of points corresponding to
        the mesh grid produced by :tvals: and :xvals:
        """
        return np.array([[self((t, x)) for x in xvals] for t in tvals])
    
    @classmethod
    def gen(cls):
        """Generates a random solution of this form, and a random location
        around which to evaluate it.
        """
        raise NotImplementedError

    
class Peakon(SolutionBase):
    """Simplest solution to the k=0 Camassa-Holm equation.
    
    Represents a soliton with a peak.
    """
    
    def __init__(self, c, **kwargs):
        """Peakons have precisely one parameter :c: defining their height and 
        speed. (We could also add an additional parameter defining their initial
        location, for consistency with TwoPeakon, but that's essentially
        unnecessary)
        """
        self.c = c
        super(Peakon, self).__init__(**kwargs)
        
    def __call__(self, point):
        t, x = point
        return self.c * np.exp(-1 * np.abs(x - self.c * t))
    
    @classmethod
    def gen(cls):
        c = np.random.uniform(3, 10)
        self = cls(c=c)
        # Random location near the peak
        t = np.random.uniform(0, 10)
        x = c * t + np.random.uniform(-2, 2)
        return (t, x), self
    

class TwoPeakon(SolutionBase):
    """Represents two solitons with peaks! Nonlinear effects start determining
    it location and magnitude.
    """
    
    def __init__(self, x1, x2, p1, p2, **kwargs):
        """TwoPeakons have essentially four parameters defining them.

        :x1: and :x2: are the initial locations of the peakons.
        :p1: and :p2: are the initial heights of the peakons. They must be
            positive. (Not zero!)
        """
        if x1 > x2:
            x1, x2 = x2, x1
            p1, p2 = p2, p1        
        
        X1 = np.exp(x1)
        X2 = np.exp(x2)
        a = p1 * p2 * (X1 - X2)
        b = (p1 + p2) * X2
        c = -X2
        discrim = np.sqrt(b ** 2 - 4 * a * c)
        twice_a = 2 * a
        
        l1 = (-b + discrim) / twice_a
        l2 = (-b - discrim) / twice_a
        
        a1 = (-1  + l2 * p2) * X2 / (p2 * (-l1 + l2))
        a2 = X2 - a1
        
        self.l1 = l1
        self.l2 = l2
        self.a1 = a1
        self.a2 = a2
        
        super(TwoPeakon, self).__init__(**kwargs)
        
    def __call__(self, point):
        t, x = point
        
        a1 = self.a1 * np.exp(t / self.l1)
        a2 = self.a2 * np.exp(t / self.l2)
        l1 = self.l1
        l2 = self.l2
        
        a1l1 = a1 * l1
        a2l2 = a2 * l2
        a1l1l1 = a1l1 * l1
        a2l2l2 = a2l2 * l2
        a1_a2 = a1 + a2
        tmp1 = a1l1l1 + a2l2l2
        tmp2 = a1l1 + a2l2
        
        # Can probably optimise this some more in terms of factorising out the
        # t dependence
        p1 = tmp1 / (l1 * l2 * tmp2)
        p2 = a1_a2 / tmp2
        x1 = np.log((a1l1l1 * a2 + a2l2l2 * a1 - 2 * a1l1 * a2l2) / tmp1)
        x2 = np.log(a1_a2)
        
        first_peakon = p1 * np.exp(-1 * np.abs(x - x1))
        second_peakon = p2 * np.exp(-1 * np.abs(x - x2))
        return first_peakon + second_peakon
    
    @classmethod
    def gen(cls):
        p1 = np.random.uniform(3, 10)
        p2 = np.random.uniform(3, 10)
        x1 = np.random.uniform(0, 3)
        x2 = np.random.uniform(3.001, 6)
        self = cls(x1, x2, p1, p2)
        # Random location near both of the peaks
        t = np.random.uniform(0, 0.5)
        left = min(x1 - 0.5 + p1 * t, x2 - 0.5 + p2 * t)
        right = max(x1 + 0.5 + p1 * t, x2 + 0.5 + p2 * t)
        middle = (right + left) / 2
        semidist = (right - left) / 2
        x = middle + semidist * np.random.uniform(-1, 1) ** 3
        return (t, x), self
    
# And for more peakons the algebra gets disgusting (and more importantly, slow), so 
# we'll leave it at two peakons for exact solutions.


### (Feature, label) generation

def sol_on_grid(point, solution):
    """Returns the values of the :solution: on fine and coarse grids around the
    specified :point:.
    """
    # Grids at the location
    cg = grid.coarse_grid(point)
    fg = grid.fine_grid(point)
    # Features: the solution on the coarse grid
    X = solution.on_grid(cg)
    # Labels: the solution on the fine grid
    y = solution.on_grid(fg)
    return X, y


def sol_at_point(point, solution):
    """Returns the values of the :solution: on a coarse grid and at a random
    point near the specified :point:.
    """
    
    cg = grid.coarse_grid(point)
    
    # Random offset from the random location that we ask for predictions at. The
    # distribution is asymmetric because we're moving relative to :point:, which
    # is in the _bottom left_ of the central cell of the coarse grid. The asymmetric
    # distribution thus makes this relative to te centre of the central cell.
    #
    # This value is not scaled relative to the size of the grid as we expect
    # that the predictions should be scale invariant, and we do not want the
    # network to unnecessarily learn the size of coarse_grid_sep.
    x_offset = np.random.uniform(-0.5, 1.5)
    t_offset = np.random.uniform(-0.5, 1.5)
    
    # Features: the solution on the coarse grid and the point to interpolate at.
    X = solution.on_grid(cg, extra=2)
    # We tell the network the offset; as the network has no way of knowing the
    # location of the grid then adding a translation would only confuse it.
    X[-2] = t_offset - 0.5  # -0.5 to normalise
    X[-1] = x_offset - 0.5  # -0.5 to normalise
    # (Yeah, it's a bit hacky to just add them on like this. Really I should
    # convert everything over to having X be a dictionary...)
    
    t, x = point
    # Label: the solution at the interpolation point
    y = np.full(1, solution((t + t_offset * grid.coarse_grid_sep.t, 
                             x + x_offset * grid.coarse_grid_sep.x)))
    
    return X, y


def gen_one_peakon_on_grid():
    """Returns a (feature, label) pair, where the features are the values of
    a single-peakon solution on a coarse grid, and the labels are the values of
    the single-peakon solution on a fine grid.
    """
    point, peakon = Peakon.gen()
    return sol_on_grid(point, peakon)


def gen_two_peakon_on_grid():
    """Returns a (feature, label) pair, where the features are the values of
    a two-peakon solution on a coarse grid, and the labels are the values of
    the two-peakon solution on a fine grid.
    """
    point, twopeakon = TwoPeakon.gen()
    return sol_on_grid(point, twopeakon)


def gen_peakons_on_grid():
    """Returns a (feature, label) pair, where the features are the values of
    either a one-peakon or a two-peakon solution on a coarse grid, and the 
    labels are the values of the peakon solution on a fine grid.
    """
    return tools.random_function(gen_one_peakon_on_grid, gen_two_peakon_on_grid)


def gen_one_peakon_at_point():
    """Returns a (feature, label) pair, where the features are the values of
    a single-peakon solution on a coarse grid and the location of a particular
    point, and the label is the value of the single-peakon solution at that
    point.
    """
    point, peakon = Peakon.gen()
    return sol_at_point(point, peakon)


def gen_two_peakon_at_point():
    """Returns a (feature, label) pair, where the features are the values of
    a two-peakon solution on a coarse grid and the location of a particular
    point, and the label is the value of the two-peakon solution at that
    point.
    """
    point, twopeakon = TwoPeakon.gen()
    return sol_at_point(point, twopeakon)


def gen_peakons_at_point():
    """Returns a (feature, label) pair, where the features are the values of
    either a one-peakon or a two-peakon solution on a coarse grid, and the 
    location of a particular point, and the label is the value of the 
    peakon solution at that point.
    """
    return tools.random_function(gen_one_peakon_at_point, gen_two_peakon_at_point)


# A particularly nice X, y that is right on the peak of the peakon
X_peak = np.array([0.71136994, 0.64367414, 0.58242045, 0.52699581, 0.47684553,
                   0.43146768, 0.3904081 , 0.35325586, 1.53965685, 1.39313912,
                   1.26056441, 1.14060584, 1.03206285, 0.93384908, 0.84498159,
                   0.76457096, 3.33236346, 3.01524715, 2.72830845, 2.46867557,
                   2.23375003, 2.02118061, 1.82883985, 1.65480272, 7.21241639,
                   6.52606422, 5.9050271 , 5.34308947, 4.83462728, 4.37455167,
                   3.95825804, 3.58157998, 3.81911647, 4.22077645, 4.66467938,
                   5.155268  , 5.69745227, 6.29665855, 6.95888391, 7.69075612,
                   1.76455206, 1.95013162, 2.15522875, 2.38189614, 2.63240234,
                   2.90925451, 3.21522348, 3.55337148, 0.81527861, 0.90102221,
                   0.99578354, 1.10051101, 1.21625277, 1.34416719, 1.48553448,
                   1.64176951, 0.37668439, 0.41630063, 0.46008335, 0.50847074,
                   0.56194707, 0.62104756, 0.68636371, 0.75854921])
y_peak = np.array([5.34308947, 5.28992485, 5.23728921, 5.18517732, 5.13358394,
                   5.08250393, 5.03193217, 4.98186361, 4.93229323, 4.8832161 ,
                   4.83462728, 5.77198627, 5.71455405, 5.65769329, 5.6013983 ,
                   5.54566345, 5.49048318, 5.43585196, 5.38176433, 5.32821488,
                   5.27519826, 5.22270916, 6.23531118, 6.1732688 , 6.11184375,
                   6.05102988, 5.99082113, 5.93121147, 5.87219493, 5.81376561,
                   5.75591767, 5.69864534, 5.64194287, 6.73582778, 6.66880518,
                   6.60244946, 6.53675399, 6.4717122 , 6.40731759, 6.34356371,
                   6.2804442 , 6.21795273, 6.15608307, 6.09482902, 7.27652151,
                   7.20411891, 7.13243673, 7.0614678 , 6.99120502, 6.92164137,
                   6.85276989, 6.78458369, 6.71707595, 6.65023993, 6.58406894,
                   7.58429925, 7.66052273, 7.70496678, 7.62830108, 7.55239822,
                   7.4772506 , 7.40285071, 7.32919112, 7.25626445, 7.18406341,
                   7.11258079, 7.0207356 , 7.09129517, 7.16256387, 7.23454883,
                   7.30725726, 7.38069641, 7.45487364, 7.52979637, 7.60547208,
                   7.68190835, 7.68351697, 6.49904846, 6.56436498, 6.63033795,
                   6.69697395, 6.76427966, 6.8322618 , 6.90092717, 6.97028264,
                   7.04033515, 7.11109169, 7.18255935, 6.01612613, 6.0765892 ,
                   6.13765994, 6.19934444, 6.26164889, 6.32457951, 6.38814259,
                   6.45234449, 6.51719163, 6.58269049, 6.64884763, 5.56908812,
                   5.62505839, 5.68159116, 5.7386921 , 5.79636692, 5.85462137,
                   5.9134613 , 5.97289257, 6.03292114, 6.093553  , 6.15479423,
                   5.155268  , 5.2070793 , 5.25941132, 5.31226928, 5.36565848,
                   5.41958424, 5.47405197, 5.5290671 , 5.58463515, 5.64076167,
                   5.69745227])


### Batching Data


class BatchData:
    """Multithreading wrapper around tf.data.Dataset."""
    
    num_processes = 8
    
    def __init__(self, gen_one_data, batch_size=1, queue_size=50, 
                 X_dtype=None, y_dtype=None, X_shape=None, y_shape=None):
        
        self.batch_size = batch_size
        self.queue = mp.Queue(maxsize=queue_size)
        
        if any([i is None] for i in (X_dtype, y_dtype, X_shape, y_shape)):
            X, y = gen_one_data()
            X_dtype = X.dtype
            y_dtype = y.dtype
            X_shape = X.shape
            y_shape = y.shape
            self.queue.put((X, y))
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.X_shape = X_shape
        self.y_shape = y_shape
            
        def _gen_one_data():
            while True:
                while True:
                    X, y = gen_one_data()
                    # Quick check to make sure that the data is nonconstant, otherwise
                    # most preprocessing (scaling) won't work.
                    # And really, do you need a neural network to tell you the
                    # interpolated values if your input data is constant?
                    if np.max(X) - np.min(X) > 0.01 and np.max(y) - np.min(y) > 0.01:
                        break
                    else:
                        tflog.info("BatchData: Got bad data; retrying.")
                self.queue.put((X, y))
                
        self.processes = [mp.Process(target=_gen_one_data)
                          for _ in range(self.num_processes)]
        for process in self.processes:
            process.start()
        
    def __call__(self):
        def generator():
            while True:
                yield self.queue.get()
        ds = tfd.Dataset.from_generator(generator, (self.X_dtype, self.y_dtype),
                                        (self.X_shape, self.y_shape))
        ds_ = ds.batch(self.batch_size)
        return ds_
    
    def terminate(self):
        for process in self.processes:
            process.terminate()
            
    @classmethod
    def context(cls, *args, **kwargs):
        class _BatchDataContext:
            def __enter__(self_context):
                self = cls(*args, **kwargs)
                self_context.instance = self
                return self
            
            def __exit__(self_context, exc_type, exc_val, exc_tb):
                self_context.instance.terminate()
        return _BatchDataContext()
                
                

#     @staticmethod
#     def to_dataset(data):
#         """Returns a tf.data.Dataset which endlessly repeats :data:."""
#         # Lambda wrapper is because in order to be part of the same graph as
#         # the DNN, it has to be called later on.
#         return lambda: tfd.Dataset.from_tensors(data).repeat()
    
#     @classmethod
#     def batch(cls, gen_one_data, batch_size=1):
#         """Takes a function :gen_one_data: which returns a generator and a
#         :batch_size:, and returns a batch of that size. Its return value is
#         not wrapped in a tf.data.Dataset.
#         """
#         (pool, gen_data_pool, batch_list, 
#          Xdtype, ydtype, 
#          X_batch_shape, y_batch_shape) = cls._batch_setup(gen_one_data, 
#                                                           batch_size)
#         return cls._batch(pool, gen_data_pool, batch_list, Xdtype, ydtype,
#                           X_batch_shape, y_batch_shape)
        
#     @classmethod
#     def _batch_setup(cls, gen_one_data, batch_size):
#         """Here we handle setting things up for generating batches. We do
#         this once here so that it doesn't all need to be done every time.
#         (Although finding the type and shape of X and y is the only truly
#         slow part.)
#         """
        
#         # Generating data is _slow_, because in general we have to use Python
#         # to do so. So we have to use tf.py_func to feed that into TensorFlow
#         # (this is what is called by tf.data.Dataset.from_generator). But
#         # tf.py_func only runs its function inside the one and only Python
#         # interpreter that is has access to, namely the one that it is called
#         # from itself.
#         # So in order to achieve speedup via multiprocessing, we have to do
#         # multiprocessing the Python way rather than the TensorFlow way.
#         pool = cls._get_pool()
#         # Now we have this strange looking partial of a global function. When
#         # we come to generate our data later, this is the function that we'll
#         # be calling. _gen_one_data is a global function because the default
#         # Python multiprocessing package is only capable of pickling top-level
#         # functions. It then has to be a partial of this function so that we
#         # can pass it gen_one_data, i.e. the function that we're actually
#         # calling. The redundant _ argument in _gen_one_data is the the value
#         # from the iterable batch_list, which is necessary to tell the
#         # multiprocessing map how many times we want to call the function.
#         gen_data_pool = ft.partial(_gen_one_data, gen_one_data=gen_one_data)
#         batch_list = list(range(batch_size))
        
#         # Call the function once so we know what its size and type is.
#         X, y = gen_one_data()
#         Xdtype = X.dtype
#         ydtype = y.dtype
#         X_batch_shape = (batch_size, *X.shape)
#         y_batch_shape = (batch_size, *y.shape)
        
#         return (pool, gen_data_pool, batch_list, Xdtype, ydtype, 
#                 X_batch_shape, y_batch_shape)
        
#     @staticmethod
#     def _batch(pool, gen_data_pool, batch_list, Xdtype, ydtype, 
#                X_batch_shape, y_batch_shape):
#         """Actually generates a batch of data using the objects supplied
#         to it from _batch_setup.
#         """
        
#         # We vectorize our data generation. Note that we have to do this
#         # this here (and not via the Dataset.batch method), because we're
#         # doing Python multiprocessing, not TensorFlow multiprocessing:
#         # and we're using multiprocessing to generate multiple elements
#         # of a batch simulataneously.
#         results = pool.map(gen_data_pool, batch_list)
#         X_batch = np.empty(X_batch_shape, dtype=Xdtype)
#         y_batch = np.empty(y_batch_shape, dtype=ydtype)
#         X_batch[:], y_batch[:] = zip(*results)
        
#         return X_batch, y_batch
