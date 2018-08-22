"""Provides for generating and batching data, and converting it into the
tf.data.Dataset format expected by TensorFlow.
"""

import functools as ft
import multiprocessing as mp
import numpy as np
import os

import tensorflow as tf
tfd = tf.data

# https://github.com/patrick-kidger/tools
import tools

from . import data_gen_base as dgb
from . import exceptions as ex
from . import fenics as fc
from . import grid


### Solutions to the Camassa--Holm equation
    
class Peakon(dgb.SolutionBase):
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
        t = 0
        x = np.random.uniform(-2, 2)
        return (t, x), self
    

class TwoPeakon(dgb.SolutionBase):
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
        # Biased towards the middle of the region
        x = middle + semidist * np.random.uniform(-1, 1) ** 3
        return (t, x), self
    
# And for more peakons the algebra gets disgusting, so we'll leave it at two 
# peakons for exact solutions.

        
class GenGeneralSolution:
    """Wraps the three different way of creating solutions that we have: 
    namely one peakon, two peakon, and FEniCS.
    """
    
    def __init__(self, num_fenics=5, num_two_peakon=4, num_one_peakon=1, 
                 fenics_from_save=False, **kwargs):
        """May be passed :fenics_num:, :two_peakon_num: and :one_peakon_num:
        arguments, which specify the proportion (relative to each other) 
        that each type of solution should be created.
        """
        
        if num_fenics > 0:
            # Not using dependency inversion, creating a 'GenSolutionFactory' 
            # is OTT ravioli code for this problem.
            self.fenics_solution_repeater = fc.FenicsSolutionRepeater(**kwargs)
            self.gen_functions = [self.fenics_solution_repeater] * num_fenics
        else:
            self.gen_functions = []
        self.gen_functions += [TwoPeakon.gen] * num_two_peakon
        self.gen_functions += [Peakon.gen] * num_one_peakon
        
        self.fenics_from_save = fenics_from_save
        
    def thread_prepare(self, thread, max_thread):
        np.random.seed(thread)
        if self.fenics_from_save:
            if hasattr(self, 'fenics_solution_repeater'):
                self.fenics_solution_repeater.thread_prepare(thread, max_thread)
        
    def __call__(self):
        return tools.random_function(*self.gen_functions)
    
    
class GenSolutionBase:
    def __init__(self, **kwargs):
        self.gen_solution = GenGeneralSolution(**kwargs)
        
    @staticmethod
    def sol_func(point, solution):
        raise NotImplementedError
        
    def thread_prepare(self, thread, max_thread):
        self.gen_solution.thread_prepare(thread, max_thread)
        
    def __call__(self):
        while True:
            point, solution = self.gen_solution()
            X, y = self.sol_func(point, solution)
            # Quick check to make sure that the data is nonconstant, otherwise
            # most preprocessing (scaling) won't work.
            # And really, do you need a neural network to tell you the
            # interpolated values if your input data is constant?
            X_corners = [X[i] for i in grid.coarse_grid_center_indices]
            # Ideally we'd have customised checking for each type of
            # preprocessing but this will do for now
            if np.max(X_corners) - np.min(X_corners) > 0.01:
                break
        return X, y

    
class GenSolutionGrid(GenSolutionBase):
    """Calls to instances return a (feature, label) pair, where the features 
    are the values of either a single-peakon, a two-peakon, or a FEniCS 
    solution on a coarse grid, and the labels are the values of the solution
    on a fine grid.
    """
    
    @staticmethod
    def sol_func(point, solution):
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
         
        
class GenSolutionPoint(GenSolutionBase):
    """Calls to instances return a (feature, label) pair, where the features 
    are the values of either a single-peakon, a two-peakon, or a FEniCS 
    solution on a coarse grid, and the location of a particular point,
    and the labels are the values of the solution on a fine grid.
    """
    
    @staticmethod
    def sol_func(point, solution):
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


### Multithreaded generating and batching of data

class BatchData:
    """Multithreading wrapper around tf.data.Dataset. Note that its
    terminate() method should be called when the instance is finished
    with.
    """
    
    def __init__(self, gen_one_data, batch_size=1, queue_size=50, 
                 X_dtype=None, y_dtype=None, X_shape=None, y_shape=None,
                 num_processes=None):
        """Initialising this class will create a queue of length :queue_size:
        and start populating it with the return values from :gen_one_data:.
        The argument :num_processes: determines how many processes will be
        used to call :gen_one_data:. (Note then that calls of :gen_one_data:
        might return their results out of order with the order that they were
        called in.) It defaults to the result of os.cpu_count().
        
        
        The argument :batch_size: is used to determine the size of the
        batches that it later produces. The arguments :X_dtype:, :y_dtype:,
        :X_shape: and :y_shape: should be the dtypes and shapes of the
        features (X) and labels (y) produced by gen_one_data. If any of them
        are set to None (their default), then :gen_one_data: will be called
        once to determine them automatically.
        """
        
        self.batch_size = batch_size
        self.queue = mp.Queue(maxsize=queue_size)

        if any([i is None] for i in (X_dtype, y_dtype, X_shape, y_shape)):
            X, y = gen_one_data()
            X_dtype = X.dtype
            y_dtype = y.dtype
            X_shape = X.shape
            y_shape = y.shape
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.X_shape = X_shape
        self.y_shape = y_shape

        def _gen_one_data(thread, max_thread):
            def gen_one_data_wrapper():
                gen_one_data.thread_prepare(thread, max_thread)
                while True:
                    self.queue.put(gen_one_data())
            return gen_one_data_wrapper
                
        if num_processes is None:
            num_processes = os.cpu_count()
        self.processes = [mp.Process(target=_gen_one_data(i, num_processes))
                          for i in range(num_processes)]
        
        for process in self.processes:
            process.start()
            
        self.terminated = False
        
    def __call__(self):
        """Creates a tf.data.Dataset gives batches of the appropriate size.
        """
        
        if not self.terminated:
            def generator():
                while True:
                    yield self.queue.get()
            # As we want a Dataset that keeps producing (feature, label) pairs
            # forever, we have to use the from_generator constructor. (I don't
            # think any of the others allow for online data production like this.)
            ds = tfd.Dataset.from_generator(generator, (self.X_dtype, self.y_dtype),
                                            (self.X_shape, self.y_shape))
            return ds.batch(self.batch_size)
        else:
            raise ex.TerminatedBatchData
    
    def terminate(self):
        """Terminates the processes that this instance uses."""
        for process in self.processes:
            process.terminate()
        self.terminated = True
            
    @classmethod
    def context(cls, *args, **kwargs):
        """For use in with statements. Creates a BatchData and automatically
        terminates it afterwards.
        """
        
        class _BatchDataContext:
            def __enter__(self_context):
                self = cls(*args, **kwargs)
                self_context.instance = self
                return self
            
            def __exit__(self_context, exc_type, exc_val, exc_tb):
                self_context.instance.terminate()
                
        return _BatchDataContext()
    
    @classmethod
    def batch(cls, gen_one_data, batch_size=1):
        """Takes a function :gen_one_data: which returns a generator and a
        :batch_size:, which defaults to 1, and returns a batch of that size. 
        Its return value is not wrapped in a tf.data.Dataset.
        """
        
        with cls.context(gen_one_data=gen_one_data) as self:
            X_batch = []
            y_batch = []
            for _ in range(batch_size):
                X, y = self.queue.get()
                X_batch.append(X)
                y_batch.append(y)
            return np.array(X_batch), np.array(y_batch)
        
    @staticmethod
    def to_dataset(data):
        """Returns a tf.data.Dataset which endlessly repeats :data:."""
        # Lambda wrapper is because in order to be part of the same graph as
        # the DNN, so it has to be called later on.
        return lambda: tfd.Dataset.from_tensors(data).repeat()
