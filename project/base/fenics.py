"""Provides for the generation of numerical solutions via 
FEniCS.
"""

import itertools as it
import math
import numpy as np
import fenics as fc

import tensorflow as tf
tflog = tf.logging

# https://github.com/patrick-kidger/tools
import tools

from . import data_gen as dg
from . import exceptions as ex
from . import grid


def _create_function_spaces(t, T, a, b, fineness_t, fineness_x):
    """Here we create the function spaces involved (and also the
    mesh as well).
    """
    
    nx = round((b - a) / fineness_x)  # number of space steps

    ### Define periodic boundary
    class PeriodicBoundary(fc.SubDomain):

        def inside(self, x, on_boundary):
            return bool(x[0] < fc.DOLFIN_EPS and 
                        x[0] > -fc.DOLFIN_EPS and 
                        on_boundary)

        # Map right boundary to left boundary
        def map(self, x, y):
            y[0] = x[0] - (b - a)

    ### Create mesh and define function spaces
    mesh = fc.IntervalMesh(nx, a, b)
    F_ele = fc.FiniteElement("CG", mesh.ufl_cell(), 1)
    V = fc.FunctionSpace(mesh, F_ele, 
                         constrained_domain=PeriodicBoundary())
    W = fc.FunctionSpace(mesh, fc.MixedElement([F_ele, F_ele]), 
                         constrained_domain=PeriodicBoundary())
    
    return V, W, mesh
    
    
def _find_initial_conditions(initial_condition, V, a, b):
    """We determine the initial condition for m, and interpolate to
    find the initial condition for u.
    """
    
    u0_uninterpolated = fc.Expression(initial_condition, degree=1)
    u0 = fc.interpolate(u0_uninterpolated, V)

    # Find initial value m0 for m
    q = fc.TestFunction(V)
    m = fc.TrialFunction(V)

    dx = fc.dx
    am = q * m * dx
    Lm = (q * u0 + q.dx(0) * u0.dx(0)) * dx

    m0 = fc.Function(V)
    fc.solve(am == Lm, m0)
    
    return m0, u0


def _create_variational_problem(m0, u0, W, dt):
    """We set up the variational problem."""
    
    p, q = fc.TestFunctions(W)
    w = fc.Function(W)  # Function to solve for
    m, u = fc.split(w)
    # Relabel i.e. initialise m_prev, u_prev as m0, u0.
    m_prev, u_prev = m0, u0
    m_mid = 0.5 * (m + m_prev)
    u_mid = 0.5 * (u + u_prev)
    F = (
        (q * u + q.dx(0) * u.dx(0) - q * m) * fc.dx +                                          # q part
        (p * (m - m_prev) + dt * (p * m_mid * u_mid.dx(0) - p.dx(0) * m_mid * u_mid)) * fc.dx  # p part
        )
    J = fc.derivative(F, w)
    problem = fc.NonlinearVariationalProblem(F, w, J=J)
    solver = fc.NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["maximum_iterations"] = 100  # Default is 50
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    
    return solver, w, m_prev, u_prev


def fenics_solve(initial_condition, t, T, a, b, fineness_t, fineness_x, 
                 smoothing_thresh=0.01):
    """Solves the Camassa--Holm equation numerically, from the given
    :initial_condition:, which should be a string of valid C++ code;
    see the FEniCS documentation for what may be passed to 
    fenics.Expression. Also requires start and end times :t:, :T:, and
    start and end space points :a:, :b:. The domain will be made
    spatially periodic by identifying :a: and :b:. Finally, the function
    requires argumetns :fineness_t: and :fineness_x:, specifying the
    fineness of the mesh on which to compute.
    
    Due to numerical errors it is possible for the function to develop
    negative values, despite this being mathematically wrong. This can
    potentially generate quite serious discrepancies. To prevent this,
    some smoothing around zero may optionally be performed be setting a
    value for :smoothing_thresh:, which defaults to 0.01. This may be
    disabled by setting :smoothing_thresh: to None.
    """
    
    nt = int((T - t) / fineness_t)  # number of time steps
    dt = fineness_t
    
    V, W, mesh = _create_function_spaces(t, T, a, b, fineness_t, fineness_x)
    m0, u0 = _find_initial_conditions(initial_condition, V, a, b)
    # We begin by storing the intial condition.
    uvals = [u0.compute_vertex_values(mesh)]
    
    solver, w, m_prev, u_prev = _create_variational_problem(m0, u0, W, dt)
    
    # Constant for smoothing out values close to zero
    if smoothing_thresh is not None:
        smoothing_const = 1 / (2 * np.sqrt(smoothing_thresh))
    
    ### Time step through the problem
    for n in range(nt):
        
#         if n % 50 == 0:
#             E = fc.assemble((u_prev * u_prev + u_prev.dx(0) * u_prev.dx(0)) * fc.dx)
#             print("time {:0>4} energy {}".format(t, E))  # Energy should remain constant

        t += dt
        # solver.solve() returns False (in its second argument) if it fails to converge.
        results = solver.solve()
        if not results[1]:
            return None, None, None, False
        # For some reason m_prev.assign(m) fails unless a deepcopy is made here
        m, u = w.split(deepcopy=True)
        
        # The solutions should never go negative. But due to either messiness across
        # the periodic boundary or a too-coarse grid, then occasionally small negative
        # solitons are generated.
        # So we fix that here, by performing a small smoothing across zero.
        # This means that the energy isn't _quite_ conserved, but it's better than
        # having a negative soliton messing up the rest of the solution.
        if smoothing_thresh is not None:
            u_array = np.array(u.vector())
            for i, val in enumerate(u_array):
                if val < -smoothing_thresh:
                    u_array[i] = 0.0
                elif val < smoothing_thresh:
                    u_array[i] = (smoothing_const * (val + smoothing_thresh)) ** 2
                    
#             for i, val in enumerate(u_array):
#                 if val < smoothing_thresh:
#                     u_array[i] = smoothing_thresh * np.e ** (val / smoothing_thresh - 1)

            # So really we're smoothing the coefficients of the basis vectors here, 
            # not the values of the function itself, so to speak. I don't think 
            # there's a better way to do this though.
            u.vector().set_local(u_array)
        
        uvals.append(u.compute_vertex_values(mesh))  # Save result
        m_prev.assign(m)  # Update for next loop
        u_prev.assign(u)  #
    
    uvals = np.array(uvals)
    xvals = mesh.coordinates()[:, 0].copy()
    tvals = np.linspace(t, T, nt + 1)
    
    return tvals, xvals, uvals, True


class FenicsSolution(dg.SolutionBase):
    """Generates a random solution using FEniCS."""
    
    defaults = tools.Object(t=0, T=10, a=0, b=20, 
                            fineness_t=grid.fine_grid_sep.t, 
                            fineness_x=grid.fine_grid_sep.x,
                            min_num_peaks=2, max_num_peaks=3, 
                            min_wobbly=2, max_wobbly=4,
                            wobbly_const_coef_lim=np.pi, 
                            wobbly_lin_coef_lim=1.7, 
                            peak_range_offset=0.15, 
                            peak_offset=3,
                            min_height=3, max_height=10)
    
    def __init__(self, initial_condition, 
                 t=defaults.t, T=defaults.T, 
                 a=defaults.a, b=defaults.b,
                 fineness_t=defaults.fineness_t,  
                 fineness_x=defaults.fineness_x, 
                 line_up=True,
                 smoothing_thresh=0.01,
                 **kwargs):
        """Numerically determines the solution to the Camassa--Holm equation
        from the given :initial_condition:.
        
        The :initial_condition: argument should be a string, in C++ syntax
        describing the initial condition. Some common gotchas: all explicit 
        numbers should be floats, the spatial variable should be referred to 
        as 'x[0]', and any absolute values should be applied as 'fabs'. 
        e.g.
        >>> initial_condition = '0.2 * exp(-fabs(x[0] - 10))'
        Check the FEniCS documentation for what are valid inputs to a
        fenics.Expression for a list of available mathematics functions.
        
        The arguments :t:, :T: describe the start and end times, and :a:, :b:
        describe the start and end spatial points. The points :a: and :b: will 
        be identified in order to make the domain spatially periodic. The 
        numerical analysis will be done on a grid of fineness :fineness_t: and 
        :fineness_x: in the t and x dimensions respectively.
        
        If :line_up: is True, then an additional linear function will be 
        added on to the initial condition in order to make sure that the 
        values at the spatial endpoints line up; otherwise a small jump is
        created across the periodic boundary, which creates a small soliton of
        its own!
        If the initial condition is sensible then this linear function will be 
        small enough to be unnoticable in the rest of the initial condition.
        If this flag is used then the :initial_condition: must also be 
        intpretable as Python, allowing for functions from the math library.
        The :line_up: argument defaults to True.
        
        Due to numerical errors it is possible for the function to develop
        negative values; despite this being mathematically wrong. This can
        potentially generate quite serious discrepancies. To prevent this,
        some smoothing around zero may optionally be performed be setting a
        value for :smoothing_thresh:, which defaults to 0.01. This may be
        disabled by setting :smoothing_thresh: to None.
        """
                
        # Incoming awfulness
        if line_up:
            # brace yourself
            math_list = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh', 
                         'degrees', 'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp', 
                         'hypot', 'ldexp', 'log', 'log10', 'modf', 'pi', 'pow', 
                         'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']
            math_dict = {name: getattr(math, name) for name in math_list}
            math_dict['abs'] = abs
            math_dict['xxx'] = a
            initial_condition_rep = initial_condition.replace('x[0]', 'xxx')
            a_val = eval(initial_condition_rep, {'__builtins__': None}, math_dict)
            math_dict['xxx'] = b
            b_val = eval(initial_condition_rep, {'__builtins__': None}, math_dict)
            # Blame http://lybniz2.sourceforge.net/safeeval.html for showing me
            # how to do this.

            _diff = a_val - b_val
            _m = _diff / (b - a)
            _c = -0.5 * _diff - _m * a

            linear_str = '{} * x[0] + {}'.format(_m, _c)
            line_up_message = ('FEniCS: Making solution periodic by adding {} to the '
                               'initial condition.'.format(linear_str))
            if abs(_c) > 0.25:
                tflog.warn(line_up_message)
            else:
                tflog.info(line_up_message)
            initial_condition += ' + ' + linear_str

        tvals, xvals, uvals, converged = fenics_solve(initial_condition, t, T, a, b, 
                                                      fineness_t, fineness_x,
                                                      smoothing_thresh=smoothing_thresh)

        if not converged:
            raise ex.FEniCSConvergenceException('FEniCS: Failed to converge.')
            
        self.initial_condition = initial_condition
        self.t = t
        self.T = T
        self.a = a
        self.b = b
        self.fineness_t = fineness_t
        self.fineness_x = fineness_x
        
        self.tvals = tvals
        self.xvals = xvals
        self.uvals = uvals
        
        super(FenicsSolution, self).__init__(**kwargs)
        
    def __call__(self, point):
        t, x = point
        t = int(t / self.fineness_t)
        x = int(x / self.fineness_x)
        return self.uvals[t, x]
    
    @classmethod
    def gen(cls, 
            min_num_peaks=defaults.min_num_peaks, 
            max_num_peaks=defaults.max_num_peaks, 
            min_wobbly=defaults.min_wobbly, 
            max_wobbly=defaults.max_wobbly,
            wobbly_const_coef_lim=defaults.wobbly_const_coef_lim, 
            wobbly_lin_coef_lim=defaults.wobbly_lin_coef_lim, 
            peak_range_offset=defaults.peak_range_offset, 
            peak_offset=defaults.peak_offset,
            min_height=defaults.min_height, 
            max_height=defaults.max_height, 
            **kwargs):
        """Generates a random solution of this form, and a random location
        around which to evaluate it.
        
        The arguments :min_num_peaks:, :max_num_peaks:, :min_wobbly:, :max_wobbly:,
        :min_height:, :max_height:, :peak_offset: determine the nature of the 
        automatically generated initial condition.
        It will randomly have between :min_num_peaks: and :max_num_peaks: 
        (inclusive) peaks (each a sech curve), each of a height chosen randomly
        from a uniform distribution between :min_height: and :max_height:.
        
        These peaks are then made 'wobbly' (technical term) by a factor 
        corresponding toan integer chosen randomly between :min_wobbly: and 
        :max_wobbly: (inclusive). (Specifically, it is multiplied by a sum of 
        sines of linear functions). Beyond making the initial condition more 
        interesting, these may also split a peak into pieces, giving the impression 
        that there more peaks than :max_num_peaks:. This 'wobbly' behaviour may be 
        turned off by setting :max_wobbly: to 0.
        
        The constant and linear coefficients of the linear functions fed into the 
        'wobbly' sin functions will be chosen randomly from a uniform distribution
        from -:wobbly_const_coef_lim: to :wobbly_const_coef_lim:, and chosen 
        randomly from a uniform distribution from -:wobbly_lin_coef_lim: to 
        :wobbly_lin_coef_lim:, respectively. The default for :wobbly_const_coef_lim:
        is pi; the default for :wobbly_lin_coef_lim: is 1.7.
        
        The peaks are located at least :peak_offset: distance from the endpoints
        of the domain, in order to allow sufficient decay for there not to be too
        large a jump across the periodic boundary. The default is 3.
        More than that, the peaks are each started off in their own section of the 
        domain: the domain (less the :peak_offset: distance from each endpoint) is 
        split into a number of equal size pieces equal to the number of peaks, and
        each peak started off in its own piece. The peak will be placed at least 
        :peak_range_offset: proportion within its own piece. The default is 0.15,
        so the peak is started off somewhere in the middle 70% of its range.
        
        The default arguments have all been chosen to try and generate interesting
        looking solutions, which are nonetheless not so wild that the numerical
        analysis is poor, but which are also nontrivial throughout most of the
        domain, so that picking an arbitrary location in the domain is likely to
        generate good training data.
        
        Any additional kwargs (e.g. :a:, :b:) are passed on to __init__.
        """
        
        self = cls._gen_solution(min_num_peaks, max_num_peaks, 
                                 min_wobbly, max_wobbly, 
                                 wobbly_const_coef_lim, wobbly_lin_coef_lim,
                                 peak_range_offset, peak_offset,
                                 min_height, max_height, 
                                 **kwargs)
        point = cls._gen_point(**kwargs)
        
        return point, self
        
    @classmethod
    def _gen_solution(cls, 
                      min_num_peaks, max_num_peaks, 
                      min_wobbly, max_wobbly, 
                      wobbly_const_coef_lim, wobbly_lin_coef_lim,
                      peak_range_offset, peak_offset,
                      min_height, max_height, 
                      **kwargs):
        """Handles the generation of the numerical solution."""
        
        a = kwargs.get('a', cls.defaults.a)
        b = kwargs.get('b', cls.defaults.b)
        
        num_peaks = np.random.randint(min_num_peaks, max_num_peaks + 1)
        # Each peak is placed randomly in a region of this length.
        peak_region_length = (b - a - 2 * peak_offset) / num_peaks
        peak_strs = []
        for peak_index in range(num_peaks):
            peak_height = np.random.uniform(min_height, max_height)
            peak_region_start = a + peak_offset + (peak_index + peak_range_offset) * peak_region_length
            peak_loc = np.random.uniform(peak_region_start, 
                                         peak_region_start + (1 - 2 * peak_range_offset) * peak_region_length)

            if max_wobbly > 0:
                wobbly = np.random.randint(min_wobbly, max_wobbly + 1)
                const_coefs = (np.random.uniform(-wobbly_const_coef_lim, wobbly_const_coef_lim) 
                               for _ in range(wobbly))
                lin_coefs = (np.random.uniform(-wobbly_lin_coef_lim, wobbly_lin_coef_lim) 
                             for _ in range(wobbly))
                wobble_strs = ('sin({} * x[0] + {})'.format(lin, const) 
                               for lin, const in zip(lin_coefs, const_coefs))
                wobble_str = ' + '.join(wobble_strs)
                norm_wobble_str = '{} + {}'.format(wobbly, wobble_str)
            else:
                wobbly = 1
                norm_wobble_str = '1.0'

            # *0.5 because norm_wobble_str takes  values in [0, 2]
            peak_strs.append('{} * ({}) / cosh(x[0] - {})'.format(0.5 * peak_height / wobbly, 
                                                                  norm_wobble_str, 
                                                                  peak_loc))

        initial_condition = ' + '.join(peak_strs)
        tflog.info("FEniCS: Generated initial condition {}".format(initial_condition))
        converged = False
        while not converged:
            try:
                self = cls(initial_condition, **kwargs)
            except ex.FEniCSConvergenceException as e:
                tflog.warn(e)
            else:
                converged = True
                
        return self
       
    @classmethod
    def _gen_point(cls, **kwargs):
        """Handles the generation of a particular point."""
        a = kwargs.pop('a', cls.defaults.a)
        b = kwargs.pop('b', cls.defaults.b)
        t = kwargs.pop('t', cls.defaults.t)
        T = kwargs.pop('T', cls.defaults.T)
        fineness_t = kwargs.pop('fineness_t', cls.defaults.fineness_t)
        fineness_x = kwargs.pop('fineness_x', cls.defaults.fineness_x)
        if kwargs != {}:
            raise TypeError("_gen_point() got unexpected keyword arguments {}".format(kwargs))
        # The coarse grid musn't have any part of it lie outside [t, T]x[a, b],
        # as we don't have any data there.
        # (Taking off all of grid.num_intervals, rather than just half of it, 
        # is a little overkill.)
        t_point = np.random.uniform(t + grid.num_intervals.t * grid.coarse_grid_sep.t, 
                                    T - grid.num_intervals.t * grid.coarse_grid_sep.t)
        x_point = np.random.uniform(a + grid.num_intervals.x * grid.coarse_grid_sep.x, 
                                    b - grid.num_intervals.x * grid.coarse_grid_sep.x)
        t_point = tools.round_mult(t_point, fineness_t, 'down')
        x_point = tools.round_mult(x_point, fineness_x, 'down')
        return t_point, x_point

    
# So from a software dev point of view, this is an interesting problem.
# I want to return the same instance of FenicsSolution several times
# over before throwing it away... which is of course very similar in
# nature to the infamous Singleton. Which isn't a problem which has
# one clear good way to handle it, to my knowledge.
# So this probably isn't the 'best' solution from a theoretical
# standpoint, but things are simple enough that it is at least the 
# easiest to implement.
class FenicsSolutionRepeater:
    """Provides for a more efficient way to use FEniCS-based solutions.
    
    As the numerical analysis with FEniCS is slow, then we want to reuse
    the same solution multiple times when generating data: we'll just
    sample different spacetime regions of the solution to generate
    different training samples.
    
    This class accepts a :repeat: argument on initialisation, specifying
    how many times a solution should be reused before another one is
    generated. Other than that argument, initialise this class with the
    arguments that FenicsSolution.gen accepts; subsequently calling this
    instance will return a point and solution just like 
    FenicsSolution.gen would.
    """
    
    defaults = FenicsSolution.defaults
    
    def __init__(self, 
                 repeat=100, 
                 min_num_peaks=defaults.min_num_peaks, 
                 max_num_peaks=defaults.max_num_peaks, 
                 min_wobbly=defaults.min_wobbly, 
                 max_wobbly=defaults.max_wobbly,
                 wobbly_const_coef_lim=defaults.wobbly_const_coef_lim, 
                 wobbly_lin_coef_lim=defaults.wobbly_lin_coef_lim, 
                 peak_range_offset=defaults.peak_range_offset, 
                 peak_offset=defaults.peak_offset,
                 min_height=defaults.min_height, 
                 max_height=defaults.max_height, 
                 **kwargs):
        self.repeat = repeat
        self.current_count = 0
        
        self.min_num_peaks = min_num_peaks
        self.max_num_peaks = max_num_peaks
        self.min_wobbly = min_wobbly
        self.max_wobbly = max_wobbly
        self.wobbly_const_coef_lim = wobbly_const_coef_lim
        self.wobbly_lin_coef_lim = wobbly_lin_coef_lim
        self.peak_range_offset = peak_range_offset
        self.peak_offset = peak_offset
        self.min_height = min_height
        self.max_height = max_height
        self.kwargs = kwargs
        
        self.solution = None
        
    def __call__(self):
        if self.current_count >= self.repeat or self.current_count == 0:
            self._gen_solution()
            self.current_count = 1
        else:
            self.current_count += 1
        point = FenicsSolution._gen_point(**self.kwargs)
        return point, self.solution
        
    
    def _gen_solution(self):
        self.solution = FenicsSolution._gen_solution(self.min_num_peaks, 
                                                     self.max_num_peaks,
                                                     self.min_wobbly, 
                                                     self.max_wobbly,
                                                     self.wobbly_const_coef_lim, 
                                                     self.wobbly_lin_coef_lim,
                                                     self.peak_range_offset, 
                                                     self.peak_offset,
                                                     self.min_height, 
                                                     self.max_height, 
                                                     **self.kwargs)

        
class GeneralSolutionGenerator:
    """Now that we have FEniCS-based solutions alongside the simple
    peakon solutions, we need a wrapper around all of them that will
    randomly give us a (point, solution) pair from any of them.
    
    This is that wrapper.
    """
    def __init__(self, fenics_num=5, two_peakon_num=4, one_peakon_num=1, 
                 **kwargs):
        # Screw dependency inversion, creating a 
        # 'GeneralSolutionGeneratorFactory' is OTT ravioli code for
        # this problem. (Things are already looking a bit ravioli-ish
        # as it is!)
        if fenics_num > 0:
            self.fenics_solution_repeater = FenicsSolutionRepeater(**kwargs)
            self.gen_functions = [self.fenics_solution_repeater] * fenics_num
        else:
            self.gen_functions = []
        self.gen_functions += [dg.TwoPeakon.gen] * two_peakon_num
        self.gen_functions += [dg.Peakon.gen] * one_peakon_num
        
    def __call__(self):
        return tools.random_function(*self.gen_functions)

    
class GenGeneralSolutionOnGrid:
    """Aaaand now that we have a wrapper around all of solution
    generators that gives us a random solution, we still need to be able
    to evaluate those solutions on a grid. Which is what this does.
    """
    def __init__(self, **kwargs):
        self.solution_generator = GeneralSolutionGenerator(**kwargs)
        
    def __call__(self):
        point, solution = self.solution_generator()
        return dg.sol_on_grid(point, solution)
        
        
class GenGeneralSolutionAtPoint:
    """...And we need to evaluate them at a point too."""
    
    def __init__(self, **kwargs):
        self.solution_generator = GeneralSolutionGenerator(**kwargs)
        
    def __call__(self):
        point, solution = self.solution_generator()
        return dg.sol_at_point(point, solution)
