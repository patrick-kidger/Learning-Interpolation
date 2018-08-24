"""Provides some base objects for generating data; to then be used when
generating both peakon and FEniCS data.
"""

import numpy as np


class SolutionBase:
    """An abstract base class for solutions to the Camassa-Holm equation.
    """
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
