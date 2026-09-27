"""Operator 1: the antiderivative -- the "hello world" of operator learning.

    G(u)(y) = integral_0^y u(s) ds,        y in [0, 1]

Input functions u are GRF samples (grf.py). G is LINEAR in u, so it is the
easiest possible operator: the exact answer has the form
sum_k b_k(u) t_k(y) with b_k LINEAR functionals of u (an SVD of the
integration kernel). A plain DeepONet reaches ~1e-2 relative test error.

The same grid is used for the m sensors (branch input) and the Q query
points (trunk input) -- convenient, but not required: the trunk can be
evaluated at ANY y after training.

"Truth" is computed with the cumulative trapezoidal rule on that grid.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from grf import sample_grf


def generate(n_samples, m=100, length_scale=0.2, seed=0):
    """Return (x, U, Y, S).

    x : (m,)          sensor locations
    U : (n, m)        input functions at the sensors  -> branch input
    Y : (m, 1)        query points                    -> trunk input
    S : (n, m)        G(u_i)(y_j), the targets
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, m)
    U = sample_grf(x, n_samples, length_scale, rng=rng)
    S = cumulative_trapezoid(U, x, axis=1, initial=0.0)
    return x, U, x[:, None], S


if __name__ == "__main__":
    x, U, Y, S = generate(5)
    print(f"U: {U.shape}, Y: {Y.shape}, S: {S.shape}")
