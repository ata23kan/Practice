"""Example 2: learn the heat-equation solution operator u0 -> u(x, t).

What is new compared with the antiderivative example:
  * the trunk input is 2D, y = (x, t): time is just another coordinate, and
    the whole space-time solution comes out of one forward pass (no time
    stepping);
  * the output data has a known, exact rank (= number of sine modes in u0),
    visible as a cliff in the singular value plot. p must sit to the right
    of that cliff, exactly like choosing the rank r of a POD basis.

Try n_modes = 20: the error gets several times worse even though the
operator is still linear and p = 40 > 20. At t = 0 the trunk must now draw
sin(20 pi x), and a small tanh network of (x, t) learns high frequencies
very slowly ("spectral bias", the same effect you see in PINNs).

Usage:
    python run_heat.py
Figures are written to DeepONet/figures/.
"""

import os

import numpy as np
import torch

np.seterr(all="ignore")  # spurious BLAS warnings, see run_antiderivative.py

from data_heat1d import generate
from deeponet import DeepONet
from plotting import plot_history, plot_output_spectrum, plot_spacetime, plot_time_slices
from training import evaluate, predict, to_tensor, train

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    torch.manual_seed(0)

    # 1. Data. Y holds (x, t) pairs of the full 64 x 41 space-time grid.
    m, n_modes = 50, 10
    x_sensors, U_train, Y, S_train, x, t = generate(1000, m=m, n_modes=n_modes, seed=0)
    _, U_test, _, S_test, _, _ = generate(200, m=m, n_modes=n_modes, seed=1)
    print(f"train: {U_train.shape[0]} functions, test: {U_test.shape[0]} functions, "
          f"{m} sensors, {Y.shape[0]} space-time query points")

    # 2. Model: branch R^m -> R^p, trunk R^2 -> R^p
    p = 40
    model = DeepONet(branch_sizes=[m, 100, 100, p], trunk_sizes=[2, 100, 100, 100, p])
    n_params = sum(w.numel() for w in model.parameters())
    print(f"DeepONet with p = {p} basis functions, {n_params} parameters")

    # 3. Train
    U_train_t, S_train_t, Y_t, U_test_t, S_test_t = to_tensor(
        U_train, S_train, Y, U_test, S_test)
    history = train(model, U_train_t, S_train_t, Y_t, U_test_t, S_test_t,
                    n_iters=20000, batch_size=100, lr=2e-3, decay_every=4000)

    # 4. Evaluate on unseen initial conditions
    test_err = evaluate(model, U_test_t, S_test_t, Y_t)
    print(f"test relative L2 error: mean={test_err.mean():.2e}, max={test_err.max():.2e}")

    # Reshape one test sample back to the (n_t, n_x) space-time grid for plotting
    i = 0
    s_true = S_test[i].reshape(len(t), len(x))
    s_pred = predict(model, U_test_t[i:i + 1], Y_t)[0].reshape(len(t), len(x))

    # --- Plots ---
    plot_history(history, os.path.join(FIG_DIR, "heat_history.png"))
    plot_output_spectrum(S_train, p, os.path.join(FIG_DIR, "heat_output_spectrum.png"))
    plot_spacetime(x, t, s_true, s_pred, os.path.join(FIG_DIR, "heat_spacetime.png"))
    plot_time_slices(x, t, s_true, s_pred, os.path.join(FIG_DIR, "heat_time_slices.png"))

    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
