"""Operator 2: solution operator of the 1D heat equation, in space AND time.

    u_t = nu * u_xx,     x in (0, 1),  t in (0, T]
    u(0, t) = u(1, t) = 0,     u(x, 0) = u0(x)

Operator:  G : u0  ->  u(x, t)   on the whole space-time slab [0,1] x [0,T].
So the trunk input is 2D, y = (x, t), and time is just another coordinate:
one forward pass gives the solution at any (x, t), no time stepping.

Input functions are random sine series (a GRF written directly in its KL
form, see grf.py):

    u0(x) = sum_{k=1}^K c_k sin(k pi x),     c_k ~ N(0, 1) / k

and the exact solution is available in closed form (separation of variables):

    u(x, t) = sum_{k=1}^K c_k exp(-nu (k pi)^2 t) sin(k pi x)

Note what this says about the OUTPUT set: every solution lies in the span
of the K functions exp(-nu (k pi)^2 t) sin(k pi x). The output data matrix
therefore has rank exactly K, and a DeepONet with p >= K basis functions can
in principle represent it perfectly. run_heat.py plots this rank cliff.
"""

import numpy as np


def generate(n_samples, m=50, n_x=64, n_t=41, nu=0.02, t_end=1.0,
             n_modes=10, seed=0):
    """Return (x_sensors, U0, Y, S, x, t).

    x_sensors : (m,)            sensor locations for u0
    U0        : (n, m)          initial conditions at the sensors -> branch input
    Y         : (n_x * n_t, 2)  query points (x, t), x fastest    -> trunk input
    S         : (n, n_x * n_t)  u(x, t) at the query points, the targets
    x, t      : (n_x,), (n_t,)  the space and time grids behind Y (for plots)
    """
    rng = np.random.default_rng(seed)
    k = np.arange(1, n_modes + 1)
    c = rng.standard_normal((n_samples, n_modes)) / k  # sine coefficients

    x_sensors = np.linspace(0.0, 1.0, m)
    U0 = c @ np.sin(np.pi * np.outer(k, x_sensors))

    x = np.linspace(0.0, 1.0, n_x)
    t = np.linspace(0.0, t_end, n_t)
    X, T = np.meshgrid(x, t)                            # (n_t, n_x) each
    Y = np.column_stack([X.ravel(), T.ravel()])         # (n_t * n_x, 2)

    # basis[k, j] = exp(-nu (k pi)^2 t_j) sin(k pi x_j) at query point j
    basis = np.exp(-nu * np.outer((k * np.pi) ** 2, Y[:, 1])) \
        * np.sin(np.pi * np.outer(k, Y[:, 0]))
    S = c @ basis

    return x_sensors, U0, Y, S, x, t


if __name__ == "__main__":
    x_sensors, U0, Y, S, x, t = generate(5)
    print(f"U0: {U0.shape}, Y: {Y.shape}, S: {S.shape}")
