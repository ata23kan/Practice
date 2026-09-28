"""Example 1: 2D heat equation, learn (u0, nu) -> u(., T) for a SCALAR diffusivity.

Train on nu in [0.02, 0.08], then test on unseen nu from 0.005 to 0.64
(4x below to 8x above the training range).

What to look at:
  * heat2d_nu_error_vs_nu.png -- low error inside the training range (green),
    fast growth outside it. The network has only learned how the solution
    depends on nu over the range it has seen.
  * the SEMIGROUP trick for large nu. The heat flow satisfies

        exp(nu T Lap) = [ exp((nu/m) T Lap) ]^m,

    i.e. diffusing with nu for time T is the same as diffusing with nu/m m
    times in a row. So for nu > nu_hi we pick m = ceil(nu / nu_hi) and apply
    the FNO m times with nu/m, which IS in the training range. This is
    extrapolation by exact physics, not by the network. There is no such
    trick for small nu (you cannot "un-compose" a diffusion), so below the
    range the error stays high.
  * heat2d_nu_transfer.png -- the operator is a Fourier multiplier
    exp(-nu |k|^2 T). We measure the multiplier the FNO actually applies and
    compare. Beyond the kept FNO modes, only the pointwise path W carries
    information.

Usage:
    python run_heat2d_nu.py
Figures are written to FNO/figures/.
"""

import os
import time

import numpy as np
import torch

from data_heat2d import T, generate, make_inputs
from fno import FNO2d
from plotting import (plot_error_vs_parameter, plot_fields, plot_history,
                      plot_transfer_functions, transfer_function)
from training import DEVICE, predict, relative_l2, train

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
NU_TRAIN = (0.02, 0.08)
N_GRID = 64
MODES = 12


def to_device(array):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=DEVICE)


def predict_nu(model, u0, nu, n_apply=1):
    """Apply the FNO n_apply times with nu / n_apply (semigroup property)."""
    u = to_device(u0)
    for _ in range(n_apply):
        A = to_device(make_inputs(u.cpu().numpy(), np.full(len(u), nu / n_apply), NU_TRAIN))
        u = predict(model, A)[:, 0]
    return u


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    torch.manual_seed(0)
    print(f"device: {DEVICE}")

    # 1. Data: exact solutions, nu log-uniform in the training range
    u0_tr, nu_tr, u_tr = generate(1000, N_GRID, NU_TRAIN, seed=0)
    u0_te, nu_te, u_te = generate(200, N_GRID, NU_TRAIN, seed=1)
    A_train, U_train = to_device(make_inputs(u0_tr, nu_tr, NU_TRAIN)), to_device(u_tr[:, None])
    A_test, U_test = to_device(make_inputs(u0_te, nu_te, NU_TRAIN)), to_device(u_te[:, None])
    print(f"train: {len(u0_tr)} samples, test: {len(u0_te)} samples, grid {N_GRID}^2, "
          f"nu in {NU_TRAIN}")

    # 2. Model: 2 input channels [u0, s(nu)] -> 1 output channel u(T)
    model = FNO2d(in_channels=2, width=32, modes=MODES, n_layers=4, homogeneous=True).to(DEVICE)
    print(f"FNO: {sum(p.numel() for p in model.parameters())} parameters "
          f"(complex weights count once)")

    # 3. Train
    t0 = time.time()
    history = train(model, A_train, U_train, A_test, U_test, epochs=150, batch_size=20, lr=1e-3)
    print(f"training time: {time.time() - t0:.0f} s")

    # 4. Unseen diffusivities, inside and outside the training range
    nu_sweep = np.geomspace(0.005, 0.64, 15)
    err_direct, err_semi = [], []
    print(f"\n{'nu':>8} {'in range':>9} {'direct':>10} {'m':>3} {'semigroup':>10}")
    for i, nu in enumerate(nu_sweep):
        u0, _, u = generate(100, N_GRID, nu=nu, seed=100 + i)
        u_true = to_device(u)
        e_direct = relative_l2(predict_nu(model, u0, nu), u_true).mean().item()
        m = int(np.ceil(nu / NU_TRAIN[1] - 1e-9))
        e_semi = relative_l2(predict_nu(model, u0, nu, m), u_true).mean().item() if m > 1 else np.nan
        err_direct.append(e_direct)
        err_semi.append(e_semi)
        inside = NU_TRAIN[0] <= nu <= NU_TRAIN[1]
        print(f"{nu:8.4f} {'yes' if inside else 'no':>9} {e_direct:10.2e} {m:3d} {e_semi:10.2e}")

    # 5. Resolution invariance: same model, finer grid, no retraining
    for n in (48, 128):
        u0, nu, u = generate(100, n, NU_TRAIN, seed=7)
        A = to_device(make_inputs(u0, nu, NU_TRAIN))
        e = relative_l2(predict(model, A)[:, 0], to_device(u)).mean().item()
        print(f"grid {n:3d}^2 (trained on {N_GRID}^2): mean relative L2 error {e:.2e}")

    # --- Plots ---
    plot_history(history, os.path.join(FIG_DIR, "heat2d_nu_history.png"))
    plot_error_vs_parameter(nu_sweep, err_direct, err_semi, NU_TRAIN, "diffusivity nu",
                            os.path.join(FIG_DIR, "heat2d_nu_error_vs_nu.png"))

    cases = [("below range", 0.005), ("in range", 0.04), ("above range", 0.32)]
    rows, labels, curves = [], [], []
    for label, nu in cases:
        u0, _, u = generate(100, N_GRID, nu=nu, seed=3)
        u_pred = predict_nu(model, u0, nu).cpu().numpy()
        rows.append([u0[0], u[0], u_pred[0], np.abs(u_pred[0] - u[0])])
        labels.append(f"nu = {nu} ({label})")
        K, H = transfer_function(u0, u_pred)
        curves.append((f"nu = {nu} ({label})", nu * T, K, H))
    plot_fields(rows, labels, ["u0", "u(T) exact", "u(T) FNO", "|error|"],
                os.path.join(FIG_DIR, "heat2d_nu_fields.png"))
    plot_transfer_functions(curves, MODES, os.path.join(FIG_DIR, "heat2d_nu_transfer.png"))
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
