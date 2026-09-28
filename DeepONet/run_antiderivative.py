"""Example 1: learn the antiderivative operator G(u)(y) = int_0^y u(s) ds.

The smallest complete DeepONet workflow:
  1. sample random input functions u, compute their outputs G(u)
  2. build branch (sees u at m sensors) and trunk (sees y)
  3. train on some functions, test on functions never seen in training
  4. look at the predictions AND at the learned trunk basis functions

Why 30000 iterations for such an easy operator? The m = 100 sensor values of
a smooth random function are strongly correlated, so the branch input is
badly conditioned and gradient descent converges slowly along its
low-variance directions. With a FROZEN, exact basis in the trunk (POD modes
of the outputs), a branch that is a single linear layer -- a convex least-
squares problem -- still needs thousands of Adam steps. The slow part of
training is the branch, not the trunk.

Usage:
    python run_antiderivative.py
Figures are written to DeepONet/figures/.
"""

import os

import numpy as np
import torch

# macOS Accelerate BLAS emits spurious divide/overflow warnings inside
# matmul (same issue as in ROM/); results were verified to be correct.
np.seterr(all="ignore")

from data_antiderivative import generate
from deeponet import DeepONet
from plotting import plot_basis, plot_history, plot_input_output_pairs
from training import evaluate, predict, to_tensor, train

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    torch.manual_seed(0)

    # 1. Data: 1000 training functions, 200 test functions (different seeds,
    # so the test functions are genuinely new).
    m = 100
    x, U_train, Y, S_train = generate(1000, m=m, seed=0)
    _, U_test, _, S_test = generate(200, m=m, seed=1)
    print(f"train: {U_train.shape[0]} functions, test: {U_test.shape[0]} functions, "
          f"{m} sensors, {Y.shape[0]} query points")

    # 2. Model: branch R^m -> R^p, trunk R^1 -> R^p
    p = 32
    model = DeepONet(branch_sizes=[m, 64, 64, p], trunk_sizes=[1, 64, 64, p])
    n_params = sum(w.numel() for w in model.parameters())
    print(f"DeepONet with p = {p} basis functions, {n_params} parameters")

    # 3. Train
    U_train_t, S_train_t, Y_t, U_test_t, S_test_t = to_tensor(
        U_train, S_train, Y, U_test, S_test)
    history = train(model, U_train_t, S_train_t, Y_t, U_test_t, S_test_t,
                    n_iters=30000, batch_size=100, lr=2e-3, decay_every=6000)

    # 4. Evaluate on unseen functions
    test_err = evaluate(model, U_test_t, S_test_t, Y_t)
    print(f"test relative L2 error: mean={test_err.mean():.2e}, max={test_err.max():.2e}")

    # The trunk is a continuous function of y, so we can query the learned
    # operator on a grid 5x finer than anything used in training.
    y_fine = np.linspace(0.0, 1.0, 500)[:, None]
    (y_fine_t,) = to_tensor(y_fine)
    S_pred_fine = predict(model, U_test_t[:3], y_fine_t)
    with torch.no_grad():
        T_fine = model.basis(y_fine_t).numpy()

    # --- Plots ---
    plot_history(history, os.path.join(FIG_DIR, "antiderivative_history.png"))
    plot_input_output_pairs(
        x, U_test[:3], {"true": x, "pred": y_fine[:, 0]}, S_test[:3], S_pred_fine,
        test_err[:3], os.path.join(FIG_DIR, "antiderivative_predictions.png"),
        output_label="int_0^y u(s) ds")
    plot_basis(y_fine[:, 0], T_fine, os.path.join(FIG_DIR, "antiderivative_trunk_basis.png"))

    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
