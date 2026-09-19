"""End-to-end demo: nonlinear (naive) Galerkin ROM for viscous Burgers 1D.

Same generic pod.py / galerkin_rom.py machinery as the heat case -- only
the FOM (fom_burgers1d.py) changes. POD is computed via the covariance /
method-of-snapshots route here (cross-checked against direct SVD) instead
of direct SVD, to exercise both.

Usage:
    python run_burgers_galerkin.py
Figures are written to ROM/figures/.
"""

import os

import numpy as np

# See run_heat_galerkin.py: spurious BLAS warnings on tiny transient values.
np.seterr(all="ignore")

from fom_burgers1d import simulate
from galerkin_rom import (build_mean_offset, build_reduced_linear_operator,
                           integrate_rom, reconstruct, reduced_initial_condition)
from pod import choose_rank, compute_pod, energy_captured
from plotting import plot_error, plot_singular_values, plot_snapshot_comparison, relative_error

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Generate FOM snapshot data
    x, t, U, A, nonlinear_term = simulate()
    print(f"FOM snapshots: {U.shape[0]} DOFs x {U.shape[1]} time samples")

    # 2. POD via the covariance / method-of-snapshots route (N=400 > M=300
    # here, so compute_pod eigendecomposes the smaller 300x300 temporal
    # correlation matrix rather than the 400x400 spatial covariance).
    mean, Phi_full, sigma = compute_pod(U, subtract_mean=True, method="covariance")

    # Cross-check against direct SVD: singular values must agree (signs of
    # individual modes may differ -- that's a harmless basis convention).
    _, _, sigma_svd = compute_pod(U, subtract_mean=True, method="svd")
    max_sigma_mismatch = np.max(np.abs(sigma[:20] - sigma_svd[:20]))
    print(f"covariance vs SVD singular value mismatch (first 20): {max_sigma_mismatch:.2e}")

    r = choose_rank(sigma, threshold=0.9999)
    print(f"Chosen rank r = {r} captures {100 * energy_captured(sigma, r):.4f}% energy")

    Phi = Phi_full[:, :r]

    # 3. Build and integrate the reduced system. The convection term is
    # nonlinear, so it's evaluated at full order each RHS call (naive
    # Galerkin, no hyper-reduction) -- no analytic jac passed here since
    # the reduced Jacobian is no longer just Ar.
    Ar = build_reduced_linear_operator(A, Phi)
    offset = build_mean_offset(A, mean, Phi)
    a0 = reduced_initial_condition(U[:, 0], mean, Phi)
    a = integrate_rom(Phi, mean, Ar, offset, t, a0, nonlinear_term=nonlinear_term)
    U_rom = reconstruct(mean, Phi, a)

    # 4. Compare FOM vs ROM
    err = relative_error(U, U_rom)
    print(f"relative error: mean={err.mean():.2e}, max={err.max():.2e}")

    # --- Plots ---
    plot_singular_values(sigma, os.path.join(FIG_DIR, "burgers_singular_values.png"))
    plot_snapshot_comparison(x, t, U, U_rom, r,
                              os.path.join(FIG_DIR, "burgers_fom_vs_rom_snapshots.png"))
    plot_error(t, err, r, os.path.join(FIG_DIR, "burgers_rom_error.png"))

    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
