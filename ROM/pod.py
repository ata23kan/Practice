"""Proper Orthogonal Decomposition (POD) of a snapshot matrix.

Given snapshots U in R^(N x M) (N = spatial DOFs, M = number of time
samples), POD finds the orthonormal basis that best captures the variance
in U, in the least-squares sense. Two equivalent routes are provided:

* "svd": direct SVD of the (mean-subtracted) snapshot matrix,
      U_fluct = Phi @ diag(sigma) @ Psi^T
  Phi's columns (left singular vectors) are the POD modes.

* "covariance": POD modes are the eigenvectors of the covariance matrix
      C = U_fluct @ U_fluct^T  (N x N, spatial covariance)
  since C = Phi @ diag(sigma^2) @ Phi^T is exactly the eigendecomposition
  of C -- this is the statistical (PCA) definition of POD. Whenever
  N > M (far more grid points than snapshots, the usual case), forming
  the N x N covariance matrix is wasteful, so we instead eigendecompose
  the much smaller M x M "temporal correlation" matrix
      K = U_fluct^T @ U_fluct
  and map its eigenvectors back to spatial modes (the "method of
  snapshots", Sirovich 1987) -- mathematically identical result, far
  cheaper when M << N.

Both methods return the same singular values/modes up to sign; "svd" is
numerically the more robust default, "covariance" is provided to make the
PCA/covariance connection explicit and to reproduce it when needed.
"""

import numpy as np


def compute_pod(U, subtract_mean=True, method="svd"):
    """Compute POD modes and singular values of snapshot matrix U.

    Parameters
    ----------
    U : ndarray, shape (N, M)
    subtract_mean : if True, decompose U = mean[:, None] + fluctuations
        and run POD on the fluctuations only (Reynolds-decomposition style).
    method : "svd" (direct SVD) or "covariance" (eigendecomposition of the
        smaller Gram matrix -- spatial N x N or temporal M x M).

    Returns
    -------
    mean : ndarray, shape (N,) -- zero vector if subtract_mean is False
    Phi : ndarray, shape (N, r_max) -- POD modes (columns), ordered by energy
    sigma : ndarray, shape (r_max,) -- singular values, descending
    """
    mean = U.mean(axis=1) if subtract_mean else np.zeros(U.shape[0])
    U_fluct = U - mean[:, None]

    if method == "svd":
        Phi, sigma, _ = np.linalg.svd(U_fluct, full_matrices=False)
    elif method == "covariance":
        Phi, sigma = _pod_via_covariance(U_fluct)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'svd' or 'covariance'")

    return mean, Phi, sigma


def _pod_via_covariance(U_fluct):
    """POD modes/singular values via eigendecomposition of a Gram matrix.

    Picks whichever of the spatial covariance (N x N) or temporal
    correlation (M x M) matrix is smaller, per the method of snapshots.
    """
    N, M = U_fluct.shape

    if N <= M:
        C = U_fluct @ U_fluct.T  # spatial covariance, N x N
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        Phi = eigvecs[:, order]
        sigma = np.sqrt(eigvals)
    else:
        K = U_fluct.T @ U_fluct  # temporal correlation, M x M
        eigvals, eigvecs = np.linalg.eigh(K)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]
        sigma = np.sqrt(eigvals)

        Phi = np.zeros((N, M))
        nonzero = sigma > 1e-12 * sigma[0]
        Phi[:, nonzero] = (U_fluct @ eigvecs[:, nonzero]) / sigma[nonzero]

    return Phi, sigma


def energy_captured(sigma, r):
    """Fraction of total snapshot energy captured by the first r modes."""
    energy = sigma**2
    return energy[:r].sum() / energy.sum()


def choose_rank(sigma, threshold=0.9999):
    """Smallest r such that the first r modes capture `threshold` energy."""
    energy = sigma**2
    cumulative = np.cumsum(energy) / energy.sum()
    return int(np.searchsorted(cumulative, threshold) + 1)
