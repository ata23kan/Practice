"""Where naive Galerkin ROMs break: low-viscosity Burgers, fixed rank.

The main run_burgers_galerkin.py always picks r via an energy threshold, so
it silently adds modes to compensate as the problem gets harder -- it never
shows failure. Here we FIX r and lower nu instead, which is what actually
exposes the classic Galerkin-ROM instability for advection-dominated
problems: truncated modes normally drain energy from the resolved scales
(numerical/physical dissipation at the grid/mode cutoff); without them the
resolved coefficients can rack up spurious energy, especially right at a
steep gradient. This is precisely the gap that closure models (including
PINN-based ones) are built to patch.

Usage:
    python run_burgers_low_viscosity.py
Figures are written to ROM/figures/.
"""

import os

import numpy as np

np.seterr(all="ignore")

from fom_burgers1d import simulate
from galerkin_rom import (build_mean_offset, build_reduced_linear_operator,
                           integrate_rom, reconstruct, reduced_initial_condition)
from pod import compute_pod, energy_captured
from plotting import relative_error
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def run_rom(x, t, U, A, nonlinear_term, mean, Phi_full, r):
    Phi = Phi_full[:, :r]
    Ar = build_reduced_linear_operator(A, Phi)
    offset = build_mean_offset(A, mean, Phi)
    a0 = reduced_initial_condition(U[:, 0], mean, Phi)
    a = integrate_rom(Phi, mean, Ar, offset, t, a0, nonlinear_term=nonlinear_term)
    return reconstruct(mean, Phi, a)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    n, t_end, r_fixed = 800, 1.5, 6

    # 1. Error growth vs. viscosity at a FIXED, deliberately small rank.
    nus = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    mean_errs, overshoot = [], []
    for nu in nus:
        x, t, U, A, N = simulate(n=n, nu=nu, t_end=t_end, n_snapshots=300)
        mean, Phi_full, sigma = compute_pod(U, subtract_mean=True, method="covariance")
        U_rom = run_rom(x, t, U, A, N, mean, Phi_full, r_fixed)
        err = relative_error(U, U_rom)
        mean_errs.append(err.mean())
        overshoot.append(np.nanmax(np.abs(U_rom)) / np.max(np.abs(U)))
        print(f"nu={nu:.5f}  r={r_fixed}  energy={100*energy_captured(sigma, r_fixed):.2f}%  "
              f"mean_err={err.mean():.3e}  overshoot={overshoot[-1]:.2f}x")

    fig, ax1 = plt.subplots()
    ax1.loglog(nus, mean_errs, "o-", color="tab:blue", label="mean relative error")
    ax1.set_xlabel("viscosity nu")
    ax1.set_ylabel("mean relative L2 error", color="tab:blue")
    ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.semilogx(nus, overshoot, "s--", color="tab:red", label="amplitude overshoot")
    ax2.set_ylabel("max|U_rom| / max|U_fom|", color="tab:red")
    ax1.set_title(f"Naive Galerkin ROM degradation at fixed r={r_fixed}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "burgers_lowvisc_error_vs_nu.png"), dpi=150)
    plt.close(fig)

    # 2. Direct visualization: same low-viscosity case, increasing rank.
    nu = 0.0005
    x, t, U, A, N = simulate(n=n, nu=nu, t_end=t_end, n_snapshots=300)
    mean, Phi_full, _ = compute_pod(U, subtract_mean=True, method="covariance")
    ranks = [6, 15, 40]
    idx = int(0.6 * (len(t) - 1))

    fig, axes = plt.subplots(1, len(ranks), figsize=(4.2 * len(ranks), 3.8), sharey=True)
    for ax, r in zip(axes, ranks):
        U_rom = run_rom(x, t, U, A, N, mean, Phi_full, r)
        ax.plot(x, U[:, idx], label="FOM")
        ax.plot(x, U_rom[:, idx], "--", label=f"ROM r={r}")
        ax.set_title(f"r = {r}")
        ax.set_xlabel("x")
        ax.legend()
    axes[0].set_ylabel("u(x, t)")
    fig.suptitle(f"nu = {nu}, t = {t[idx]:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "burgers_lowvisc_rank_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
