"""Everything we do is on a grid."""

import numpy as np

# https://github.com/patrick-kidger/tools
import tools


### Grid hyperparameters
# The separation between points of the fine grid
fine_grid_sep = tools.Object(t=0.01, x=0.01)
# The separation between points of the coarse grid
coarse_grid_sep = tools.Object(t=0.1, x=0.1)
# The amount of intervals in the coarse grid. Thus the coarse grid will contain
# (num_intervals.t + 1) * (num_intervals.x + 1) elements.
# So with num_intervals.t = 3, num_intervals.x = 3, it looks like:
#
# @ @ @ @
#
# @ @ @ @
#
# @ @ @ @
#
# @ @ @ @
num_intervals = tools.Object(t=7, x=7)


fine_grid_fineness = tools.Object(t=int(coarse_grid_sep.t // fine_grid_sep.t), 
                                  x=int(coarse_grid_sep.x // fine_grid_sep.x))
coarse_grid_size = tools.Object(t=num_intervals.t * coarse_grid_sep.t,
                                x=num_intervals.x * coarse_grid_sep.x)


### Grids to evaluate our solution on

def grid(point, grid_size, grid_fineness):
    """Creates a grid whose bottom left entry is at the specified :point:
    location. The size of the overall grid may be specified via :grid_size:, and
    the fineness of the subdivision by :grid_fineness:, both of which should be
    of the form tools.Object(t, x). Thus the resulting grid has
    (grid_fineness.t + 1) * (grid_fineness.x + 1) elements."""
    t, x = point
    return [(t_, x_) for t_ in np.linspace(t, t + grid_size.t, 
                                           grid_fineness.t + 1)
                     for x_ in np.linspace(x, x + grid_size.x, 
                                           grid_fineness.x + 1)]

def fine_grid(point):
    """Creates a fine grid whose bottom left entry is at the specified :point:
    location, with size and fineness determined by the earlier hyperparameters.
    """
    return grid(point, coarse_grid_sep, fine_grid_fineness)

def coarse_grid(point):
    """Creates a coarse grid for which the bottom left entry of its middle
    square is as the specified :t:, :x: location, with size and fineness
    determined by the earlier hyperparameters.
    """
    left_intervals_t = np.floor((num_intervals.t - 1) / 2)
    left_intervals_x = np.floor((num_intervals.x - 1) / 2)
    
    left_amount_t = left_intervals_t * coarse_grid_sep.t
    left_amount_x = left_intervals_x * coarse_grid_sep.x
    
    t, x = point
    bottomleft_point = (t - left_amount_t, x - left_amount_x)
    return grid(bottomleft_point, coarse_grid_size, num_intervals)

def _index_tol(cg, point, tol=0.001):
    """Searches through a list of 2-tuples, :cg:, to find the first element 
    which is within tolerance :tol: of :point:. Essentially the index method
    for lists, except this one makes sense for high precision floating point
    numbers.
    """

    t, x = point
    for i, element in enumerate(cg):
        t2, x2 = element
        if max(np.abs(t - t2), np.abs(x - x2)) < tol:
            return i
    raise ValueError('{} is not in {}'.format(point, type(cg)))
    
_cg = coarse_grid((0, 0))
coarse_grid_center_indices = tuple(_index_tol(_cg, point) for point in 
                                   ((0, 0), 
                                    (coarse_grid_sep.t, 0), 
                                    (0, coarse_grid_sep.x), 
                                    (coarse_grid_sep.t, coarse_grid_sep.x))
                                  )
_fg = fine_grid((0, 0))
fine_grid_center_indices = tuple(_index_tol(_fg, point) for point in 
                                 ((0, 0), 
                                  (coarse_grid_sep.t, 0), 
                                  (0, coarse_grid_sep.x), 
                                  (coarse_grid_sep.t, coarse_grid_sep.x))
                                )
