"""Example 3: learn the (nonlinear) Burgers solution operator u0 -> u(., T).

What is new compared with the heat example:
  * the operator is nonlinear, so the branch's nonlinearity is doing real
    work (for the two linear examples a linear branch would have sufficed);
  * the output data no longer has an exact low rank: compare the singular
    value plot with the heat example's sharp cliff (the drop near index 85
    is the solver's 2/3 dealiasing cutoff, not physics);
  * look at the WORST test case in burgers_predictions.png: the front has
    the right shape but sits in the wrong place. Moving features are hard
    for any method that adds up a fixed set of basis functions, whether
    it is a DeepONet or a POD ROM;
  * a plain DeepONet is noticeably less accurate here (~8% test error).
    Train and test errors are close, and 4x more training data does not
    help, so the limit is the network/optimizer, not the data.

Try NU = 0.01: fronts get steeper, the singular values decay more slowly,
and with the same data and network the test error grows to ~13%. Compare
with ROM/run_burgers_low_viscosity.py.

Usage:
    python run_burgers.py
Figures are written to DeepONet/figures/.
"""

import os

import numpy as np
import torch

np.seterr(all="ignore")  # spurious BLAS warnings, see run_antiderivative.py

from data_burgers1d import generate
from deeponet import DeepONet
from plotting import plot_history, plot_input_output_pairs, plot_output_spectrum
from training import evaluate, predict, to_tensor, train

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

NU = 0.02   # viscosity: lower it to make the problem harder
T_END = 0.3  # the operator maps u0 to u(., T_END)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    torch.manual_seed(0)

    # 1. Data: one Burgers solve per input function (all solved at once).
    n_x = 128
    print(f"solving Burgers for 1200 initial conditions (nu = {NU}, T = {T_END}) ...")
    x, U_train, Y, S_train = generate(1000, n_x=n_x, nu=NU, t_end=T_END, seed=0)
    _, U_test, _, S_test = generate(200, n_x=n_x, nu=NU, t_end=T_END, seed=1)
    print(f"train: {U_train.shape[0]} functions, test: {U_test.shape[0]} functions, "
          f"{n_x} sensors, {Y.shape[0]} query points")

    # 2. Model: branch R^n_x -> R^p, trunk R^1 -> R^p
    p = 64
    model = DeepONet(branch_sizes=[n_x, 128, 128, p], trunk_sizes=[1, 128, 128, 128, p])
    n_params = sum(w.numel() for w in model.parameters())
    print(f"DeepONet with p = {p} basis functions, {n_params} parameters")

    # 3. Train
    U_train_t, S_train_t, Y_t, U_test_t, S_test_t = to_tensor(
        U_train, S_train, Y, U_test, S_test)
    history = train(model, U_train_t, S_train_t, Y_t, U_test_t, S_test_t,
                    n_iters=30000, batch_size=100, lr=2e-3, decay_every=6000)

    # 4. Evaluate on unseen initial conditions
    test_err = evaluate(model, U_test_t, S_test_t, Y_t)
    print(f"test relative L2 error: mean={test_err.mean():.2e}, max={test_err.max():.2e}")

    # Show the best, the median and the worst test function
    order = np.argsort(test_err)
    show = [order[0], order[len(order) // 2], order[-1]]
    S_pred = predict(model, U_test_t[show], Y_t)

    # --- Plots ---
    plot_history(history, os.path.join(FIG_DIR, "burgers_history.png"))
    plot_output_spectrum(S_train, p, os.path.join(FIG_DIR, "burgers_output_spectrum.png"))
    plot_input_output_pairs(
        x, U_test[show], {"true": x, "pred": x}, S_test[show], S_pred, test_err[show],
        os.path.join(FIG_DIR, "burgers_predictions.png"),
        input_label="u0(x)", output_label="u(x, T)")

    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
