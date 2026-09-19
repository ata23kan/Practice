"""Full-order model (FOM): 1D heat equation with a moving Gaussian source.

    u_t = nu * u_xx + f(x, t),      x in (0, L), t in (0, T)
    u(0, t) = u(L, t) = 0           (homogeneous Dirichlet)
    u(x, 0) = 0

f(x, t) is a Gaussian bump whose center sweeps across the domain, so the
solution has a nonzero, non-trivial time-mean plus fluctuations riding on
top of it -- useful later for the mean-subtraction discussion.

Spatial discretization: standard second-order central differences on the
interior nodes, giving a tridiagonal operator A. Time integration: solve_ivp
(RK45) on the resulting semi-discrete ODE du/dt = A u + f(t).
"""

import numpy as np
from scipy.integrate import solve_ivp


def build_laplacian(n_interior, dx, nu):
    """Tridiagonal operator for nu * d^2/dx^2 with homogeneous Dirichlet BCs."""
    main = -2.0 * np.ones(n_interior)
    off = np.ones(n_interior - 1)
    A = (nu / dx**2) * (np.diag(main) + np.diag(off, 1) + np.diag(off, -1))
    return A


def forcing(x, t, amplitude=5.0, sigma=0.05, period=4.0, center_frac=0.3):
    """Gaussian bump whose center oscillates sinusoidally across the domain."""
    L = x[-1] - x[0] + (x[1] - x[0])  # domain length (x excludes boundary here)
    x0 = 0.5 * L + center_frac * L * np.sin(2 * np.pi * t / period)
    return amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def simulate(n_interior=199, L=1.0, nu=0.01, t_end=8.0, n_snapshots=400):
    """Run the FOM and return (x, t, U) with U shape (n_interior, n_snapshots)."""
    dx = L / (n_interior + 1)
    x = np.linspace(dx, L - dx, n_interior)  # interior grid points only
    A = build_laplacian(n_interior, dx, nu)

    def rhs(t, u):
        return A @ u + forcing(x, t)

    t_eval = np.linspace(0.0, t_end, n_snapshots)
    u0 = np.zeros(n_interior)

    sol = solve_ivp(rhs, (0.0, t_end), u0, t_eval=t_eval, method="BDF",
                     jac=A, rtol=1e-8, atol=1e-10)

    return x, sol.t, sol.y, A  # U = sol.y, shape (n_interior, n_snapshots)


if __name__ == "__main__":
    x, t, U, A = simulate()
    print(f"snapshot matrix shape: {U.shape}")
    print(f"grid points: {len(x)}, time steps: {len(t)}")
