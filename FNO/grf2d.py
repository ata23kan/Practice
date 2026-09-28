"""Random 2D input fields: band-limited periodic Gaussian random fields (GRF).

An operator is learned over a DISTRIBUTION of input functions, so we need
many random, smooth fields f(x, y) on the periodic square [0, 2 pi)^2.

On a periodic domain, a STATIONARY covariance k(x - x') is diagonalized by
Fourier modes (the eigenvectors of a circulant matrix are e^{i k.x}). So the
Karhunen-Loeve expansion of DeepONet/grf.py -- which needed an eigen-
decomposition of the covariance matrix -- becomes a Fourier series with
independent random coefficients:

    f(x) = sum_k  sqrt(lam_k) z_k e^{i k.x},     z_k ~ complex N(0, 1),
    lam_k = (|k|^2 + tau^2)^(-alpha).

This is the GRF  N(0, (-Laplacian + tau^2)^(-alpha))  used in the FNO paper.
alpha sets the smoothness (larger = smoother), 1/tau the correlation length.
The k = 0 mode is dropped (zero-mean fields), and f is scaled so that its
pointwise variance is 1.

(This is also why FNO's Fourier basis is the "POD basis" of statistically
homogeneous periodic data: the covariance eigenvectors ARE the Fourier modes.)

Band-limited: only |k_x|, |k_y| <= k_cut are kept. The field is then an exact
trigonometric polynomial. It can be evaluated on ANY n x n grid with
n > 2 k_cut, and it is the SAME function on all of them. We need this for the
resolution test (train on 64^2, test on 128^2).
"""

import numpy as np


def wavenumbers(n):
    """Integer wavenumbers of an n x n grid in rfft2 layout: kx (n, 1), ky (1, n//2+1)."""
    kx = np.fft.fftfreq(n, d=1.0 / n)[:, None]
    ky = np.fft.rfftfreq(n, d=1.0 / n)[None, :]
    return kx, ky


def sample_grf2d(n_samples, n_grid, k_cut=16, alpha=2.5, tau=3.0, rng=None):
    """Draw n_samples fields on an n_grid x n_grid grid -> (n_samples, n_grid, n_grid).

    The random coefficients are drawn on the fixed band |k| <= k_cut, so the
    same rng seed gives the same functions for every n_grid > 2 k_cut.
    """
    assert n_grid > 2 * k_cut, "grid must resolve the band: n_grid > 2 k_cut"
    rng = np.random.default_rng() if rng is None else rng

    # Coefficients on the band: kx in [-k_cut, k_cut], ky in [0, k_cut]
    kx_band = np.concatenate([np.arange(0, k_cut + 1), np.arange(-k_cut, 0)])[:, None]
    ky_band = np.arange(0, k_cut + 1)[None, :]
    lam = (kx_band**2 + ky_band**2 + tau**2) ** (-alpha)
    lam[0, 0] = 0.0  # zero mean
    z = (rng.standard_normal((n_samples, *lam.shape))
         + 1j * rng.standard_normal((n_samples, *lam.shape))) / np.sqrt(2)
    c_band = np.sqrt(lam) * z

    # Pointwise variance of the resulting real field. irfft2 counts every
    # ky > 0 column twice (it adds the conjugate half-plane): variance 2 lam.
    # A ky = 0 column only contributes its real part: variance lam / 2.
    weight = np.where(ky_band > 0, 2.0, 0.5)
    c_band /= np.sqrt(np.sum(weight * lam))

    # Place the band into the rfft2 array of the requested grid and invert.
    # The factor n^2 undoes irfft2's 1/n^2, so values do not depend on n.
    c = np.zeros((n_samples, n_grid, n_grid // 2 + 1), dtype=complex)
    c[:, :k_cut + 1, :k_cut + 1] = c_band[:, :k_cut + 1]
    c[:, -k_cut:, :k_cut + 1] = c_band[:, k_cut + 1:]
    return np.fft.irfft2(c, s=(n_grid, n_grid)) * n_grid**2
