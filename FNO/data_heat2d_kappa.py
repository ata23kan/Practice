"""Operator 2: 2D heat equation with a diffusivity FIELD kappa(x, y).

    u_t = div( kappa(x, y) grad u ),   (x, y) in [0, 2 pi)^2 periodic,   u(., 0) = u0.

Operator to learn:   G : (u0, kappa) -> u(., T).

This models heat conduction in a heterogeneous material. Fourier modes are no
longer eigenfunctions, there is no closed form, and G is NONLINEAR in the
whole field kappa. This is where an FNO is actually useful: the reference
solver below needs hundreds of time steps, the FNO one forward pass.

Diffusivity fields: a lognormal field around a mean level kappa_bar,

    kappa(x) = kappa_bar * exp( sigma g(x) ),   g a smooth unit-variance GRF,

so kappa > 0 everywhere and varies by about a factor exp(4 sigma) ~ 16 over
the domain (sigma = 0.7). The mean level kappa_bar is drawn log-uniformly from
a training range; tests go outside it ("unseen diffusivity range").

Reference solver (pseudo-spectral in space, RK4 in time):

    L(u) = d_x( kappa d_x u ) + d_y( kappa d_y u ),

where every derivative is taken in Fourier space (d_x <-> i k_x) and the
product with kappa in physical space. Spectral accuracy in x means the 64^2
and 128^2 solutions agree to many digits, so the resolution test measures
the network, not the solver.

Time step: L is stiff with largest eigenvalue about kappa_max |k|_max^2, and
RK4 is stable for |lambda dt| < 2.78. We take dt = 2 / (kappa_max |k|_max^2).

FNO input channels: [u0, normalized log kappa], where the log is shifted and
scaled like the scalar-nu example so that kappa_bar in the training range
gives a field centred in [-1, 1].
"""

import numpy as np
import torch

from grf2d import sample_grf2d

T = 1.0
SIGMA = 0.7          # log-diffusivity fluctuation strength
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _wavenumbers(n, device):
    k = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    kx = k[:, None]
    ky = torch.fft.rfftfreq(n, d=1.0 / n, device=device)[None, :]
    # Zero the Nyquist wavenumbers for first derivatives (standard practice:
    # the Nyquist mode's derivative is not representable as a real field).
    kx = torch.where(kx.abs() == n // 2, torch.zeros_like(kx), kx)
    ky = torch.where(ky == n // 2, torch.zeros_like(ky), ky)
    return kx, ky


def _rhs(u, kappa, kx, ky):
    """div(kappa grad u), all derivatives spectral."""
    s = u.shape[-2:]
    u_hat = torch.fft.rfft2(u)
    ux = torch.fft.irfft2(1j * kx * u_hat, s=s)
    uy = torch.fft.irfft2(1j * ky * u_hat, s=s)
    div_hat = 1j * kx * torch.fft.rfft2(kappa * ux) + 1j * ky * torch.fft.rfft2(kappa * uy)
    return torch.fft.irfft2(div_hat, s=s)


@torch.no_grad()
def solve(u0, kappa, t=T, chunk=100):
    """u0, kappa (N, n, n) numpy -> u(., t) (N, n, n) numpy.

    Samples are sorted by kappa_max and solved in chunks, each with its own
    stable dt, so a few very diffusive samples do not slow down all others.
    """
    n = u0.shape[-1]
    kx, ky = _wavenumbers(n, DEVICE)
    k2_max = 2 * (n // 2) ** 2
    order = np.argsort(kappa.reshape(len(kappa), -1).max(axis=1))
    out = np.empty_like(u0)
    for start in range(0, len(order), chunk):
        idx = order[start:start + chunk]
        u = torch.as_tensor(u0[idx], dtype=torch.float64, device=DEVICE)
        kap = torch.as_tensor(kappa[idx], dtype=torch.float64, device=DEVICE)
        n_steps = int(np.ceil(t * kap.max().item() * k2_max / 2.0))
        dt = t / n_steps
        for _ in range(n_steps):
            k1 = _rhs(u, kap, kx, ky)
            k2 = _rhs(u + 0.5 * dt * k1, kap, kx, ky)
            k3 = _rhs(u + 0.5 * dt * k2, kap, kx, ky)
            k4 = _rhs(u + dt * k3, kap, kx, ky)
            u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[idx] = u.cpu().numpy()
    return out


def encode_log_kappa(kappa, kappa_range):
    lo, hi = kappa_range
    mid = np.sqrt(lo * hi)
    return (np.log(kappa) - np.log(mid)) / np.log(hi / mid)


def make_inputs(u0, kappa, kappa_range):
    """Stack FNO input channels: (N, 2, n, n) = [u0, encoded log kappa]."""
    return np.stack([u0, encode_log_kappa(kappa, kappa_range)], axis=1)


def generate(n_samples, n_grid=64, kappa_range=(0.02, 0.08), kappa_bar=None, seed=0):
    """Random u0, random kappa fields, and the solution u(., T).

    kappa_bar=None draws the mean level log-uniformly from kappa_range;
    a number fixes it for every sample. Returns u0, kappa, kappa_bar, u.
    """
    rng = np.random.default_rng(seed)
    u0 = sample_grf2d(n_samples, n_grid, k_cut=16, alpha=2.5, tau=3.0, rng=rng)
    g = sample_grf2d(n_samples, n_grid, k_cut=8, alpha=3.0, tau=3.0, rng=rng)
    if kappa_bar is None:
        kappa_bar = np.exp(rng.uniform(np.log(kappa_range[0]), np.log(kappa_range[1]), n_samples))
    else:
        kappa_bar = np.full(n_samples, float(kappa_bar))
    kappa = kappa_bar[:, None, None] * np.exp(SIGMA * g)
    return u0, kappa, kappa_bar, solve(u0, kappa)
