"""Operator 3: solution operator of the 1D viscous Burgers equation (NONLINEAR).

    u_t + (u^2 / 2)_x = nu * u_xx,     x in [0, 1) periodic
    u(x, 0) = u0(x)

Operator:  G : u0  ->  u(., T)     (initial condition -> solution at time T)

Unlike the antiderivative and heat examples, G is nonlinear in u0: doubling
u0 does NOT double u(., T). This is the setting where the branch net's
nonlinearity actually matters. Lowering nu makes steeper fronts: the
singular values of the output data decay more slowly (so p has to grow,
the same reason linear ROMs struggle at low viscosity, compare
ROM/run_burgers_low_viscosity.py), and the map u0 -> u(., T) itself gets
harder for the branch to learn.

Input functions u0 are periodic GRF samples (grf.py). The defaults
(nu = 0.02, T = 0.3) are chosen so that fronts are still steepening at time
T. For much larger nu*T the solution has diffused to little more than its
mean plus one sine wave by time T, and the example becomes trivial.

Solver (only used to generate data): Fourier pseudo-spectral in space with
2/3-rule dealiasing, classic explicit RK4 in time. All samples are advanced
at once as rows of one array, so generating ~1000 solutions takes seconds.
"""

import numpy as np

from grf import sample_grf


def burgers_rhs_hat(u_hat, k, nu, dealias):
    """Fourier transform of  -(u^2/2)_x + nu u_xx,  given u_hat (rows = samples)."""
    u = np.fft.irfft(u_hat * dealias, axis=1)
    flux_hat = np.fft.rfft(0.5 * u**2, axis=1) * dealias
    return -1j * k * flux_hat - nu * k**2 * u_hat


def solve_burgers(U0, nu, t_end, dt):
    """Advance every row of U0 (n_samples, n_x) to time t_end with RK4."""
    n_x = U0.shape[1]
    k = 2 * np.pi * np.fft.rfftfreq(n_x, d=1.0 / n_x)  # wavenumbers on [0, 1)
    dealias = (np.arange(len(k)) < (2.0 / 3.0) * (n_x // 2)).astype(float)

    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    u_hat = np.fft.rfft(U0, axis=1)
    for _ in range(n_steps):
        k1 = burgers_rhs_hat(u_hat, k, nu, dealias)
        k2 = burgers_rhs_hat(u_hat + 0.5 * dt * k1, k, nu, dealias)
        k3 = burgers_rhs_hat(u_hat + 0.5 * dt * k2, k, nu, dealias)
        k4 = burgers_rhs_hat(u_hat + dt * k3, k, nu, dealias)
        u_hat = u_hat + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.fft.irfft(u_hat, n=n_x, axis=1)


def generate(n_samples, n_x=128, nu=0.02, t_end=0.3, length_scale=0.2,
             dt=2e-4, seed=0):
    """Return (x, U0, Y, S).

    x  : (n_x,)        periodic grid; used for both sensors and queries
    U0 : (n, n_x)      initial conditions       -> branch input
    Y  : (n_x, 1)      query points             -> trunk input
    S  : (n, n_x)      u(x, T), the targets

    dt must respect the explicit RK4 diffusion limit, roughly
    dt < 2.8 / (nu * k_max^2) with k_max = 2 pi n_x / 3.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(n_x) / n_x
    U0 = sample_grf(x, n_samples, length_scale, periodic=True, rng=rng)
    S = solve_burgers(U0, nu, t_end, dt)
    return x, U0, x[:, None], S


if __name__ == "__main__":
    x, U0, Y, S = generate(5)
    print(f"U0: {U0.shape}, Y: {Y.shape}, S: {S.shape}")
    print(f"u0 range: [{U0.min():.2f}, {U0.max():.2f}], "
          f"u(T) range: [{S.min():.2f}, {S.max():.2f}]")
