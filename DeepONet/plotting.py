"""Shared plotting helpers for the DeepONet examples."""

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, path):
    fig, ax = plt.subplots()
    ax.semilogy(history["iter"], history["train"], label="train")
    ax.semilogy(history["iter"], history["test"], label="test (unseen functions)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean relative L2 error")
    ax.set_title("Training history")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_output_spectrum(S, p, path):
    """Singular values of the output data matrix S (N x Q), with p marked.

    With a linear trunk expansion, the best a DeepONet can do on the outputs
    is the best rank-p approximation -- so if the singular values have not
    decayed by index p, no amount of training will fix it.
    """
    sigma = np.linalg.svd(S, compute_uv=False)
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, len(sigma) + 1), sigma / sigma[0], "o-", markersize=3)
    ax.axvline(p, color="tab:red", linestyle="--", label=f"p = {p}")
    ax.set_xlim(0, min(len(sigma), 4 * p))
    ax.set_ylim(max(1e-16, sigma[min(len(sigma), 4 * p) - 1] / sigma[0] / 10), 2)
    ax.set_xlabel("index")
    ax.set_ylabel("normalized singular value")
    ax.set_title("Singular values of the output data")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_basis(y, T, path, n_show=8):
    """First n_show learned trunk basis functions t_k(y) on a 1D grid."""
    fig, ax = plt.subplots()
    for k in range(min(n_show, T.shape[1])):
        ax.plot(y, T[:, k], label=f"t_{k + 1}")
    ax.set_xlabel("y")
    ax.set_title("Learned trunk basis functions (first few)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_input_output_pairs(x_in, U, y_out, S_true, S_pred, errors, path,
                            input_label="u(x)", output_label="G(u)(y)"):
    """One row per test function: input on the left, true vs predicted output
    on the right. y_out may be a finer grid than the truth (see S_pred)."""
    n = len(U)
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.4 * n), sharex="col")
    for i in range(n):
        axes[i, 0].plot(x_in, U[i], color="gray")
        axes[i, 0].set_ylabel(input_label)
        axes[i, 1].plot(y_out["true"], S_true[i], "o", markersize=3, label="truth")
        axes[i, 1].plot(y_out["pred"], S_pred[i], "-", label="DeepONet")
        axes[i, 1].set_ylabel(output_label)
        axes[i, 1].set_title(f"relative L2 error = {errors[i]:.2e}", fontsize=9)
    axes[0, 0].set_title("input function (branch)")
    axes[0, 1].legend(fontsize=8)
    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spacetime(x, t, s_true, s_pred, path):
    """Truth / prediction / error for one space-time field s(x, t), shape (n_t, n_x)."""
    extent = [x[0], x[-1], t[0], t[-1]]
    vmax = np.abs(s_true).max()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    panels = [(s_true, "truth", "RdBu_r", -vmax, vmax),
              (s_pred, "DeepONet", "RdBu_r", -vmax, vmax),
              (np.abs(s_true - s_pred), "|error|", "magma", None, None)]
    for ax, (field, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", aspect="auto", extent=extent,
                       cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("x")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_time_slices(x, t, s_true, s_pred, path, time_fracs=(0.0, 0.1, 0.3, 1.0)):
    """Profiles of one space-time field at a few times, truth vs prediction."""
    fig, axes = plt.subplots(1, len(time_fracs), figsize=(3.6 * len(time_fracs), 3.2),
                             sharey=True)
    for ax, frac in zip(axes, time_fracs):
        j = int(round(frac * (len(t) - 1)))
        ax.plot(x, s_true[j], "o", markersize=3, label="truth")
        ax.plot(x, s_pred[j], "-", label="DeepONet")
        ax.set_title(f"t = {t[j]:.2f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
