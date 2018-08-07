"""Provides for generating numerical solutions via FEniCS."""

import itertools as it
import numpy as np
import fenics as fc

from . import data_gen as dg


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
    
    return solver, w, m_prev, u_prev


def fenics_solve(initial_condition, t, T, a, b, fineness_t, fineness_x):
    nt = round((T - t) / fineness_t)  # number of time steps
    dt = fineness_t
    
    V, W, mesh = _create_function_spaces(t, T, a, b, fineness_t, fineness_x)
    m0, u0 = _find_initial_conditions(initial_condition, V)
    # We begin by storing the intial condition.
    uvals = [u0.compute_vertex_values(mesh)]
    
    solver, w, m_prev, u_prev = _create_variational_problem(m0, u0, W, dt)
    
    ### Time step through the problem
    for n in range(nt):
#         if n % 10 == 0:
#             E = assemble((u_prev * u_prev + u_prev.dx(0) * u_prev.dx(0)) * dx)
#             print("time {:0>4} energy {}".format(t, E))  # Energy should remain constant
        t += dt
        solver.solve()
        # For some reason m_prev.assign(m) fails unless a deepcopy is made here
        m, u = w.split(deepcopy=True)
        uvals.append(u.compute_vertex_values(mesh))  # Save result
        m_prev.assign(m)  # Update for next loop
        u_prev.assign(u)  #
    
    uvals = np.array(uvals)
    xvals = mesh.coordinates()[:, 0]
    tvals = np.linspace(t, T, nt + 1)
    
    return tvals, xvals, uvals


class FenicsSolution(dg.SolutionBase):
    pass