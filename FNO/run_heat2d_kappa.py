"""Example 2: 2D heat equation with a diffusivity FIELD, learn (u0, kappa) -> u(., T).

This is the problem FNO is built for: the input is a whole field kappa(x, y)
(a heterogeneous material), the map is nonlinear in kappa, and there is no
closed-form solution -- the reference solver takes hundreds of RK4 steps.

Train on mean diffusivity levels kappa_bar in [0.02, 0.08] (every sample also
has its own random spatial pattern), then test on unseen kappa_bar from 0.005
to 0.64.

Compared with the scalar-nu example:
  * kappa enters as a real field with rich Fourier content, so the spectral
    layers can use it directly (a constant nu channel only lives in the k = 0
    mode and can only act through the nonlinearities).
  * The semigroup trick still works: u_t = div(c kappa grad u) over time T is
    the same as u_t = div(kappa grad u) over time c T, so for kappa_bar above
    the range we apply the FNO m times with kappa / m.
  * Resolution test: data are solved on 128^2 as well, and the 64^2-trained
    FNO is evaluated there without retraining.
  * Speed: FNO inference time vs the reference solver on the same inputs.

Usage:
    python run_heat2d_kappa.py
Data generation takes about 1 minute and training about 3 minutes on a laptop GPU.
Figures are written to FNO/figures/.
"""

import os
import time

import numpy as np
import torch

from data_heat2d_kappa import generate, make_inputs, solve
from fno import FNO2d
from plotting import plot_error_vs_parameter, plot_fields, plot_history
from training import DEVICE, predict, relative_l2, train

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
KAPPA_TRAIN = (0.02, 0.08)
N_GRID = 64


def to_device(array):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=DEVICE)


def predict_kappa(model, u0, kappa, n_apply=1):
    """Apply the FNO n_apply times with kappa / n_apply (semigroup property)."""
    u = to_device(u0)
    for _ in range(n_apply):
        u = predict(model, to_device(make_inputs(u.cpu().numpy(), kappa / n_apply, KAPPA_TRAIN)))[:, 0]
    return u


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    torch.manual_seed(0)
    print(f"device: {DEVICE}")

    # 1. Data from the pseudo-spectral RK4 solver
    t0 = time.time()
    u0_tr, k_tr, _, u_tr = generate(1000, N_GRID, KAPPA_TRAIN, seed=0)
    u0_te, k_te, _, u_te = generate(200, N_GRID, KAPPA_TRAIN, seed=1)
    print(f"data generation: {time.time() - t0:.0f} s")
    A_train, U_train = to_device(make_inputs(u0_tr, k_tr, KAPPA_TRAIN)), to_device(u_tr[:, None])
    A_test, U_test = to_device(make_inputs(u0_te, k_te, KAPPA_TRAIN)), to_device(u_te[:, None])

    # 2. Model: 2 input channels [u0, log kappa] -> u(T). u(T) is linear in u0
    #    for fixed kappa, so homogeneous=True is exact here too.
    model = FNO2d(in_channels=2, width=32, modes=12, n_layers=4, homogeneous=True).to(DEVICE)
    print(f"FNO: {sum(p.numel() for p in model.parameters())} parameters")

    # 3. Train
    t0 = time.time()
    history = train(model, A_train, U_train, A_test, U_test, epochs=150, batch_size=20, lr=1e-3)
    print(f"training time: {time.time() - t0:.0f} s")

    # 4. Unseen mean diffusivity levels
    kbar_sweep = np.geomspace(0.005, 0.64, 15)
    err_direct, err_semi = [], []
    print(f"\n{'kappa_bar':>9} {'in range':>9} {'direct':>10} {'m':>3} {'semigroup':>10}")
    for i, kbar in enumerate(kbar_sweep):
        u0, kappa, _, u = generate(50, N_GRID, kappa_bar=kbar, seed=100 + i)
        u_true = to_device(u)
        e_direct = relative_l2(predict_kappa(model, u0, kappa), u_true).mean().item()
        m = int(np.ceil(kbar / KAPPA_TRAIN[1] - 1e-9))
        e_semi = (relative_l2(predict_kappa(model, u0, kappa, m), u_true).mean().item()
                  if m > 1 else np.nan)
        err_direct.append(e_direct)
        err_semi.append(e_semi)
        inside = KAPPA_TRAIN[0] <= kbar <= KAPPA_TRAIN[1]
        print(f"{kbar:9.4f} {'yes' if inside else 'no':>9} {e_direct:10.2e} {m:3d} {e_semi:10.2e}")

    # 5. Resolution invariance: solve on 128^2, evaluate the 64^2-trained FNO
    u0_f, k_f, _, u_f = generate(100, 128, KAPPA_TRAIN, seed=7)
    for n, step in ((128, 1), (64, 2)):
        sl = (slice(None), slice(None, None, step), slice(None, None, step))
        A = to_device(make_inputs(u0_f[sl], k_f[sl], KAPPA_TRAIN))
        e = relative_l2(predict(model, A)[:, 0], to_device(u_f[sl])).mean().item()
        print(f"grid {n:3d}^2 (trained on {N_GRID}^2): mean relative L2 error {e:.2e}")

    # 6. Speed: reference solver vs FNO on the same 200 test inputs
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    t0 = time.time()
    solve(u0_te, k_te)
    t_solver = time.time() - t0
    t0 = time.time()
    predict(model, A_test)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    t_fno = time.time() - t0
    print(f"200 solves: reference solver {t_solver:.2f} s, FNO {t_fno:.3f} s "
          f"({t_solver / t_fno:.0f}x faster)")

    # --- Plots ---
    plot_history(history, os.path.join(FIG_DIR, "heat2d_kappa_history.png"))
    plot_error_vs_parameter(kbar_sweep, err_direct, err_semi, KAPPA_TRAIN,
                            "mean diffusivity kappa_bar",
                            os.path.join(FIG_DIR, "heat2d_kappa_error_vs_kappa.png"))

    rows, labels = [], []
    for label, kbar in (("below range", 0.005), ("in range", 0.04), ("above range", 0.32)):
        u0, kappa, _, u = generate(4, N_GRID, kappa_bar=kbar, seed=3)
        m = max(1, int(np.ceil(kbar / KAPPA_TRAIN[1] - 1e-9)))
        u_pred = predict_kappa(model, u0, kappa, m).cpu().numpy()
        rows.append([np.log10(kappa[0]), u0[0], u[0], u_pred[0], np.abs(u_pred[0] - u[0])])
        labels.append(f"kappa_bar = {kbar} ({label}" + (f", m = {m})" if m > 1 else ")"))
    plot_fields(rows, labels, ["log10 kappa", "u0", "u(T) solver", "u(T) FNO", "|error|"],
                os.path.join(FIG_DIR, "heat2d_kappa_fields.png"))
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
