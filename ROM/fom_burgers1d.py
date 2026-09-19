"""Full-order model (FOM): 1D viscous Burgers equation, periodic domain.

    u_t + u u_x = nu * u_xx,      x in [0, L) periodic,   t in (0, T)
    u(x, 0) = sin(2 pi x / L) + 0.5 sin(4 pi x / L)

The nonlinear advection term is written in conservative flux form,
u u_x = (u^2/2)_x, and discretized with central differences on the flux --
standard for a *viscous* Burgers problem where nu is large enough to keep
the (near-)shock resolved on the grid (no shock-capturing scheme needed).

Semi-discrete form:   du/dt = A u + N(u),   A = nu * d^2/dx^2 (periodic,
circulant), N(u) = -d/dx(u^2/2) (periodic, central difference).

N(u) is exposed as a plain function of the full state so the exact same
physics can be reused, unmodified, inside the (naive) Galerkin ROM.
"""

import numpy as np
from scipy.integrate import solve_ivp


def build_periodic_laplacian(n, dx, nu):
    """Circulant tridiagonal operator for nu * d^2/dx^2 with periodic BCs."""
    main = -2.0 * np.ones(n)
    off = np.ones(n)
    A = (nu / dx**2) * (np.diag(main) + np.diag(off[:-1], 1) + np.diag(off[:-1], -1))
    A[0, -1] = nu / dx**2
    A[-1, 0] = nu / dx**2
    return A


def convection_term(u, dx):
    """-(d/dx)(u^2/2), central difference, periodic BCs (np.roll wraps around)."""
    flux = 0.5 * u**2
    return -(np.roll(flux, -1) - np.roll(flux, 1)) / (2 * dx)


def initial_condition(x, L):
    return np.sin(2 * np.pi * x / L) + 0.5 * np.sin(4 * np.pi * x / L)


def simulate(n=400, L=2.0, nu=0.02, t_end=1.5, n_snapshots=300):
    """Run the FOM and return (x, t, U, A, nonlinear_term).

    nonlinear_term(u, t) -> ndarray is returned so the ROM can reuse the
    identical convection-term evaluation at full order.
    """
    dx = L / n
    x = np.linspace(0.0, L, n, endpoint=False)
    A = build_periodic_laplacian(n, dx, nu)

    def nonlinear_term(u, t):
        return convection_term(u, dx)

    def rhs(t, u):
        return A @ u + nonlinear_term(u, t)

    u0 = initial_condition(x, L)
    t_eval = np.linspace(0.0, t_end, n_snapshots)

    sol = solve_ivp(rhs, (0.0, t_end), u0, t_eval=t_eval, method="BDF",
                     rtol=1e-8, atol=1e-10)

    return x, sol.t, sol.y, A, nonlinear_term


if __name__ == "__main__":
    x, t, U, A, nonlinear_term = simulate()
    print(f"snapshot matrix shape: {U.shape}")
    print(f"grid points: {len(x)}, time steps: {len(t)}")
    print(f"u range: [{U.min():.3f}, {U.max():.3f}]")
