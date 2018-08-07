"""Provides for the generation of numerical solutions via 
FEniCS.
"""

import itertools as it
import numpy as np
import fenics as fc

import tensorflow as tf
tflog = tf.logging

from . import data_gen as dg
from . import grid


def _create_function_spaces(t, T, a, b, fineness_t, fineness_x):
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
    
    
def _find_initial_conditions(initial_condition, V):
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
    p, q = fc.TestFunctions(W)
    w = fc.Function(W)  # Function to solve for
    m, u = fc.split(w)
    # Relabel i.e. initialise m_prev, u_prev as m0, u0.
    m_prev, u_prev = m0, u0
    m_mid = 0.5 * (m + m_prev)
    u_mid = 0.5 * (u + u_prev)
    dx = fc.dx
    F = (
        (q * u + q.dx(0) * u.dx(0) - q * m) * dx +                                          # q part
        (p * (m - m_prev) + dt * (p * m_mid * u_mid.dx(0) - p.dx(0) * m_mid * u_mid)) * dx  # p part
        )
    J = fc.derivative(F, w)
    problem = fc.NonlinearVariationalProblem(F, w, J=J)
    solver = fc.NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["maximum_iterations"] = 500
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    
    return solver, w, m_prev, u_prev


def fenics_solve(initial_condition, t, T, a, b, fineness_t, fineness_x):
    nt = round((T - t) / fineness_t)  # number of time steps
    dt = fineness_t
    
    V, W, mesh = _create_function_spaces(t, T, a, b, fineness_t, fineness_x)
    m0, u0 = _find_initial_conditions(initial_condition, V)
    # We begin by storing the intial condition.
    uvals = [u0.compute_vertex_values(mesh)]
    
    solver, w, m_prev, u_prev = _create_variational_problem(m0, u0, W, dt)
    
    converged = True
    ### Time step through the problem
    for n in range(nt):
#         if n % 10 == 0:
#             E = assemble((u_prev * u_prev + u_prev.dx(0) * u_prev.dx(0)) * dx)
#             print("time {:0>4} energy {}".format(t, E))  # Energy should remain constant
        t += dt
        # solver.solve() returns False (in its second argument) if it fails to converge.
        converged = converged and solver.solve()[1]
        # For some reason m_prev.assign(m) fails unless a deepcopy is made here
        m, u = w.split(deepcopy=True)
        
        # The solutions should never go negative. But due to messiness across the
        # periodic boundary then occasionally small negative solitons are generated.
        # So we fix that here.
        u_array = np.array(u.vector())
        u_array[u_array < 0] = 0
        u.vector().set_local(u_array)  
        
        uvals.append(u.compute_vertex_values(mesh))  # Save result
        m_prev.assign(m)  # Update for next loop
        u_prev.assign(u)  #
    
    uvals = np.array(uvals)
    xvals = mesh.coordinates()[:, 0].copy()
    tvals = np.linspace(t, T, nt + 1)
    
    return tvals, xvals, uvals, converged


class FenicsSolution(dg.SolutionBase):
    """Generates a random solution using FEniCS."""
    
    def __init__(self, min_num_peaks=3, max_num_peaks=6, min_poly_deg = 2, max_poly_deg=6, 
                 min_height=3, max_height=10, t=0, T=10, a=0, b=40, peak_offset=5,
                 fineness_t=grid.fine_grid_sep.t,  fineness_x=grid.fine_grid_sep.x, 
                 initial_condition=None):
        
        converged = False
        while not converged:        
            if initial_condition is None:
                num_peaks = np.random.randint(min_num_peaks, max_num_peaks + 1)
                peak_strs = []
                for _peak in range(num_peaks):
                    peak_height = np.random.uniform(min_height, max_height)
                    peak_loc = np.random.uniform(a + peak_offset, b - peak_offset)
                    x = 'x[0] - {}'.format(float(peak_loc))
                    peak_poly_deg = np.random.randint(min_poly_deg, max_poly_deg + 1)

                    coefs = [np.random.uniform(0, 0.2 ** exponent) for exponent in range(peak_poly_deg + 1)]
                    poly_pieces = [str(coefs[0])]
                    if peak_poly_deg > 0:
                        poly_pieces.append("{} * ({})".format(coefs[1], x))
                        poly_pieces.extend(["{} * pow({}, {})".format(coef, x, i + 2) 
                                            for i, coef in enumerate(coefs[2:])])
                    poly_str = ' + '.join(poly_pieces)

                    peak_strs.append('{} * pow({}, 2) / cosh({})'.format(peak_height, poly_str, x))

                _initial_condition = ' + '.join(peak_strs)

            tflog.info("FEniCS: Generated initial condition {}".format(_initial_condition))
            tvals, xvals, uvals, converged = fenics_solve(_initial_condition, t, T, a, b, 
                                                          fineness_t, fineness_x)
            
            if not converged:
                warn_msg = 'FEniCS: Failed to converge with given initial condition.'
                if initial_condition is not None:
                    raise RuntimeError(warn_msg)
                else:
                    tflog.warn(warn_msg)
            
        self.initial_condition = _initial_condition
        self.t = t
        self.T = T
        self.a = a
        self.b = b
        self.fineness_t = fineness_t
        self.fineness_x = fineness_x
        
        self.tvals = tvals
        self.xvals = xvals
        self.uvals = uvals
        
    def __call__(self, point):
        t, x = point
        t_index = self._index_tol(tvals, t)
        x_index = self._index_tol(xvals, x)
        return uvals[t_index, x_index]
    
    @staticmethod
    def _index_tol(list_, val, tol=0.0001):
        for i, element in enumerate(list_):
            if np.abs(element - val) < tol:
                return i
        raise ValueError('{} is not in {}'.format(val, type(list_)))
