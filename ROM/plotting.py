"""Shared plotting helpers for FOM-vs-ROM comparisons, used by every case."""

import matplotlib.pyplot as plt


def plot_singular_values(sigma, path):
    fig, ax = plt.subplots()
    ax.semilogy(sigma / sigma[0], "o-", markersize=3)
    ax.set_xlabel("mode index")
    ax.set_ylabel("normalized singular value")
    ax.set_title("POD singular value decay")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_snapshot_comparison(x, t, U, U_rom, r, path, time_fracs=(0.1, 0.5, 0.9)):
    fig, axes = plt.subplots(1, len(time_fracs), figsize=(4 * len(time_fracs), 3.5),
                              sharey=True)
    for ax_i, frac in zip(axes, time_fracs):
        idx = int(frac * (len(t) - 1))
        ax_i.plot(x, U[:, idx], label="FOM")
        ax_i.plot(x, U_rom[:, idx], "--", label=f"ROM (r={r})")
        ax_i.set_title(f"t = {t[idx]:.2f}")
        ax_i.set_xlabel("x")
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error(t, err, r, path):
    fig, ax = plt.subplots()
    ax.plot(t, err)
    ax.set_xlabel("t")
    ax.set_ylabel("relative L2 error")
    ax.set_title(f"FOM vs ROM error (r={r})")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def relative_error(U, U_rom):
    """Time series of relative L2 error; 0 wherever the FOM norm itself is 0."""
    fom_norm = (U**2).sum(axis=0) ** 0.5
    err = ((U - U_rom) ** 2).sum(axis=0) ** 0.5 / fom_norm
    err[fom_norm == 0] = 0.0
    return err
