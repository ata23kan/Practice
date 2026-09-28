"""Random input functions: samples of a Gaussian random field (GRF).

An operator is learned over a DISTRIBUTION of input functions, so we need a
way to draw many random, smooth functions u(x). A zero-mean GRF is defined
by its covariance kernel k(x, x'). Two kernels are provided:

    RBF      : k(x, x') = exp( -(x - x')^2 / (2 l^2) )
    periodic : k(x, x') = exp( -(L / pi)^2 sin^2( pi (x - x') / L ) / (2 l^2) )

The periodic kernel is scaled so that it reduces to the RBF kernel when
|x - x'| << L (since (L / pi) sin(pi d / L) ~= d), so l means the same
thing in both.

The length scale l controls smoothness: small l -> rough, wiggly functions,
which are harder to learn (they need more sensors, more data, larger p).

Sampling: on the grid x, form the covariance matrix K and eigendecompose it,
K = V diag(lam) V^T. Then

    u = V diag(sqrt(lam)) z,      z ~ N(0, I)

has exactly covariance K. This is the Karhunen-Loeve (KL) expansion, i.e.
POD of the covariance -- the same eigen-machinery as ROM/pod.py, used here
to GENERATE functions instead of compressing them. (Eigendecomposition is
used instead of Cholesky because K is nearly singular for smooth kernels;
tiny negative eigenvalues from round-off are clipped to zero.)
"""

import numpy as np


def rbf_kernel(x, length_scale):
    d = x[:, None] - x[None, :]
    return np.exp(-(d**2) / (2 * length_scale**2))


def periodic_kernel(x, length_scale, period=1.0):
    d = x[:, None] - x[None, :]
    chord = (period / np.pi) * np.sin(np.pi * d / period)
    return np.exp(-(chord**2) / (2 * length_scale**2))


def sample_grf(x, n_samples, length_scale, periodic=False, rng=None):
    """Draw n_samples GRF realizations on the grid x -> shape (n_samples, len(x))."""
    rng = np.random.default_rng() if rng is None else rng
    if periodic:
        period = len(x) * (x[1] - x[0])  # x is a periodic grid without endpoint
        K = periodic_kernel(x, length_scale, period)
    else:
        K = rbf_kernel(x, length_scale)
    lam, V = np.linalg.eigh(K)
    lam = np.clip(lam, 0.0, None)
    z = rng.standard_normal((len(x), n_samples))
    return (V @ (np.sqrt(lam)[:, None] * z)).T
