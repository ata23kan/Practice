"""End-to-end demo: FOM data -> POD -> linear Galerkin ROM, FOM vs ROM comparison.

Usage:
    python run_heat_galerkin.py
Figures are written to ROM/figures/.
"""

import os

import numpy as np

# BLAS emits spurious divide/overflow warnings on the tiny (~1e-27) values
# that appear right at the start of the transient; the actual results are
# fine (verified against direct computation), so silence just those.
np.seterr(all="ignore")

from fom_heat1d import forcing, simulate
from galerkin_rom import (build_mean_offset, build_reduced_linear_operator,
                           integrate_rom, reconstruct, reduced_initial_condition)
from pod import choose_rank, compute_pod, energy_captured
from plotting import plot_error, plot_singular_values, plot_snapshot_comparison, relative_error

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Generate FOM snapshot data
    x, t, U, A = simulate()
    print(f"FOM snapshots: {U.shape[0]} DOFs x {U.shape[1]} time samples")

    # 2. POD with mean subtraction
    mean, Phi_full, sigma = compute_pod(U, subtract_mean=True, method="svd")
    r = choose_rank(sigma, threshold=0.9999)
    print(f"Chosen rank r = {r} captures {100 * energy_captured(sigma, r):.4f}% energy")

    Phi = Phi_full[:, :r]

    # 3. Build and integrate the reduced system (no nonlinear term here: it's
    # purely linear, so we can pass jac=Ar directly to help the stiff solver)
    Ar = build_reduced_linear_operator(A, Phi)
    offset = build_mean_offset(A, mean, Phi)
    a0 = reduced_initial_condition(U[:, 0], mean, Phi)
    a = integrate_rom(Phi, mean, Ar, offset, t, a0,
                       forcing=lambda tt: forcing(x, tt), jac=Ar)
    U_rom = reconstruct(mean, Phi, a)

    # 4. Compare FOM vs ROM
    err = relative_error(U, U_rom)
    print(f"relative error: mean={err.mean():.2e}, max={err.max():.2e}")

    # --- Plots ---
    plot_singular_values(sigma, os.path.join(FIG_DIR, "heat_singular_values.png"))
    plot_snapshot_comparison(x, t, U, U_rom, r,
                              os.path.join(FIG_DIR, "heat_fom_vs_rom_snapshots.png"))
    plot_error(t, err, r, os.path.join(FIG_DIR, "heat_rom_error.png"))

    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
