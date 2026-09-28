"""Operator 1: 2D heat equation with a SCALAR diffusivity nu.

    u_t = nu (u_xx + u_yy),    (x, y) in [0, 2 pi)^2 periodic,    u(., 0) = u0.

Operator to learn:   G : (u0, nu) -> u(., T).

Exact solution. Fourier modes are eigenfunctions of the Laplacian
(Lap e^{i k.x} = -|k|^2 e^{i k.x}), so each mode decays independently:

    u_hat(k, T) = exp(-nu |k|^2 T) u0_hat(k).

No solver is needed and the data are exact: every error you see is the
network's. G is LINEAR in u0 but NONLINEAR in nu -- learning the nu
dependence is the whole difficulty.

nu and T only appear as the product tau = nu T (nondimensionalization):
"a larger diffusivity" and "a later time" are the same thing. T is fixed to 1
here, so nu plays the role of tau.

Input encoding for the FNO. nu is a scalar, but an FNO eats fields, so nu is
broadcast as a CONSTANT second channel, normalized so that the training range
maps to [-1, 1]:

    s(nu) = (log nu - log nu_mid) / log(nu_hi / nu_mid),   nu_mid = sqrt(nu_lo nu_hi).

log, because nu acts inside an exponential. Values of nu outside the training
range give |s| > 1 -- inputs the network has never seen.
"""

import numpy as np

from grf2d import sample_grf2d, wavenumbers

T = 1.0


def solve_exact(u0, nu, t=T):
    """u0 (N, n, n), nu (N,) -> u(., t) (N, n, n)."""
    kx, ky = wavenumbers(u0.shape[-1])
    decay = np.exp(-np.asarray(nu)[:, None, None] * (kx**2 + ky**2) * t)
    return np.fft.irfft2(decay * np.fft.rfft2(u0), s=u0.shape[-2:])


def encode_nu(nu, nu_range):
    nu_lo, nu_hi = nu_range
    nu_mid = np.sqrt(nu_lo * nu_hi)
    return (np.log(nu) - np.log(nu_mid)) / np.log(nu_hi / nu_mid)


def make_inputs(u0, nu, nu_range):
    """Stack FNO input channels: (N, 2, n, n) = [u0, s(nu) broadcast]."""
    s = np.broadcast_to(encode_nu(nu, nu_range)[:, None, None], u0.shape)
    return np.stack([u0, s], axis=1)


def generate(n_samples, n_grid=64, nu_range=(0.02, 0.08), nu=None, seed=0):
    """Random u0 and log-uniform nu in nu_range (or a given fixed nu).

    Returns u0 (N, n, n), nu (N,), u (N, n, n).
    """
    rng = np.random.default_rng(seed)
    u0 = sample_grf2d(n_samples, n_grid, rng=rng)
    if nu is None:
        nu = np.exp(rng.uniform(np.log(nu_range[0]), np.log(nu_range[1]), n_samples))
    else:
        nu = np.full(n_samples, float(nu))
    return u0, nu, solve_exact(u0, nu)
