"""Shared plotting helpers for the FNO examples."""

import matplotlib.pyplot as plt
import numpy as np

from grf2d import wavenumbers


def plot_history(history, path):
    fig, ax = plt.subplots()
    ax.semilogy(history["epoch"], history["train"], label="train")
    ax.semilogy(history["epoch"], history["test"], label="test (unseen inputs, in range)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean relative L2 error")
    ax.set_title("Training history")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_vs_parameter(values, err_direct, err_semigroup, train_range, xlabel, path):
    """Test error against the diffusivity, with the training range shaded.

    err_semigroup may contain NaN where the semigroup trick does not apply
    (values at or below the top of the training range).
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(*train_range, color="tab:green", alpha=0.15, label="training range")
    ax.loglog(values, err_direct, "o-", label="FNO, direct")
    ax.loglog(values, err_semigroup, "s--", label="FNO, semigroup (m applications)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean relative L2 error")
    ax.set_title("Prediction for unseen diffusivities")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fields(rows, row_labels, col_titles, path):
    """Grid of 2D fields. rows[i][j] is an (n, n) array. The error column
    (last) gets its own color scale; the others share one per row."""
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.9 * n_cols, 2.7 * n_rows), squeeze=False)
    for i, row in enumerate(rows):
        vmax = max(np.abs(f).max() for f in row[1:-1])
        for j, field in enumerate(row):
            ax = axes[i, j]
            if j == n_cols - 1:
                im = ax.imshow(field.T, origin="lower", cmap="magma")
            elif j == 0:
                im = ax.imshow(field.T, origin="lower", cmap="viridis")
            else:
                im = ax.imshow(field.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10)
        axes[i, 0].set_ylabel(row_labels[i], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def transfer_function(u0, u):
    """Radially binned multiplier H(|k|) with u_hat(k) ~= H(k) u0_hat(k).

    Least-squares estimate over all samples and all modes in a shell
    |k| in [K - 1/2, K + 1/2):  H = sum Re(u_hat conj(u0_hat)) / sum |u0_hat|^2.
    For the heat equation the exact answer is exp(-nu K^2 T).
    """
    kx, ky = wavenumbers(u0.shape[-1])
    shell = np.rint(np.sqrt(kx**2 + ky**2)).astype(int)
    u0_hat, u_hat = np.fft.rfft2(u0), np.fft.rfft2(u)
    num = np.real(u_hat * np.conj(u0_hat)).sum(axis=0)
    den = (np.abs(u0_hat) ** 2).sum(axis=0)
    k_max = u0.shape[-1] // 2
    K = np.arange(1, k_max)
    H = np.array([num[shell == k].sum() / den[shell == k].sum() for k in K])
    return K, H


def plot_transfer_functions(curves, k_modes, path):
    """curves: list of (label, nu, K, H_pred). Dashed lines: exact exp(-nu K^2)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for idx, (label, nu, K, H) in enumerate(curves):
        color = f"C{idx}"
        ax.semilogy(K, np.exp(-nu * K**2), "--", color=color, alpha=0.7)
        ax.semilogy(K, np.abs(H), "o", color=color, markersize=4, label=label)
    ax.axvline(k_modes, color="gray", linestyle=":", label=f"FNO modes = {k_modes}")
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel("|k|")
    ax.set_ylabel("|H(k)|  ( u_hat = H u0_hat )")
    ax.set_title("Learned multiplier (dots) vs exact exp(-nu |k|^2 T) (dashed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
