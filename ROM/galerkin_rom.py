"""Generic Galerkin ROM machinery -- independent of the underlying physics.

FOM (semi-discrete), as general as we need across cases:

    du/dt = A u + N(u, t) + f(t)

  A       : linear operator (or None if there isn't one)
  N(u, t) : optional nonlinear term, evaluated at FULL order
  f(t)    : optional external forcing

Ansatz:   u(t) ~= mean + Phi @ a(t),   Phi^T Phi = I_r

Galerkin condition (residual orthogonal to the trial subspace):

    da/dt = (Phi^T A Phi) a + Phi^T A mean
             \\_________/      \\________/
                 Ar               offset (only nonzero if mean-subtracted)

            + Phi^T N(mean + Phi a, t) + Phi^T f(t)
               \\___________________/     \\________/
                projected nonlinear term   projected forcing

Ar and offset are (r x r) / (r,) and computed once, offline -- cheap.

The nonlinear term, if present, is handled "naively": the reduced state a
is lifted back to the full state u = mean + Phi @ a (size N), N(u, t) is
evaluated there at full order, and the result is projected back down with
Phi^T. This is mathematically consistent Galerkin projection, but it does
NOT give a nonlinear term the O(N) -> O(r) speedup -- reconstructing u and
evaluating N still costs O(N) per RHS call. Hyper-reduction methods (DEIM,
EIM, gappy POD) exist specifically to fix this; they are not implemented
here. This is also exactly the gap where a learned/physics-informed
closure model can be inserted later: replace or augment
Phi^T N(mean + Phi a, t) with a cheap surrogate acting on `a` directly.
"""

import numpy as np
from scipy.integrate import solve_ivp


def build_reduced_linear_operator(A, Phi):
    """Ar = Phi^T A Phi. Pass A=None for a problem with no linear part."""
    if A is None:
        r = Phi.shape[1]
        return np.zeros((r, r))
    return Phi.T @ A @ Phi


def build_mean_offset(A, mean, Phi):
    """Phi^T A mean -- the constant term introduced by mean-subtraction.

    Zero whenever there's no linear operator or no mean-subtraction, since
    then the term vanishes identically.
    """
    if A is None or not np.any(mean):
        return np.zeros(Phi.shape[1])
    return Phi.T @ (A @ mean)


def reduced_initial_condition(u0, mean, Phi):
    """a0 = Phi^T (u0 - mean): the reduced IC consistent with mean-subtraction.

    Not zero in general when mean-subtraction is used -- forgetting this
    silently gives a wrong-starting-point ROM.
    """
    return Phi.T @ (u0 - mean)


def integrate_rom(Phi, mean, Ar, offset, t_eval, a0,
                   nonlinear_term=None, forcing=None,
                   method="BDF", rtol=1e-8, atol=1e-10, **solve_ivp_kwargs):
    """Integrate da/dt = Ar a + offset + Phi^T N(mean + Phi a, t) + Phi^T f(t).

    Parameters
    ----------
    nonlinear_term : callable(u, t) -> ndarray, shape (N,), or None
    forcing : callable(t) -> ndarray, shape (N,), or None
    solve_ivp_kwargs : forwarded to scipy.integrate.solve_ivp (e.g. jac=Ar
        for a purely linear reduced system, to help stiff solvers).

    Returns
    -------
    a : ndarray, shape (r, len(t_eval)) -- reduced trajectory
    """
    def rhs(t, a):
        out = Ar @ a + offset
        if nonlinear_term is not None:
            u_full = mean + Phi @ a
            out = out + Phi.T @ nonlinear_term(u_full, t)
        if forcing is not None:
            out = out + Phi.T @ forcing(t)
        return out

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), a0, t_eval=t_eval,
                     method=method, rtol=rtol, atol=atol, **solve_ivp_kwargs)
    return sol.y


def reconstruct(mean, Phi, a):
    """Lift reduced coordinates back to full space: u(t) ~= mean + Phi @ a(t)."""
    return mean[:, None] + Phi @ a
