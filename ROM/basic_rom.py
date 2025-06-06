#
# Solve reduced form of du/dt = Au, u(0) = u_0
#

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys

#Parameters
N = 50
timesteps = 100
r = 10 	# reduced dimension
t0 = 0.0
t1 = 10.0

# Create a random linear system
np.random.seed(0)
A = np.random.normal(0, 0.1, size=(N, N))
A = A - 0.2*np.eye(N)

# initial condition
u0 = np.random.normal(0, 1, size=(N,))

# simualte and collect snapshots
t_eval = np.linspace(t0, t1, timesteps)

def rhs(t, u):
	return A @ u

sol = solve_ivp(rhs, [t0, t1], u0, t_eval=t_eval, method='RK45')
U = sol.y  # shape: (N, timesteps)

# Create POD Basis U = V S W^T
V, S, WT = np.linalg.svd(U, full_matrices=False)

plt.figure()
plt.semilogy(S, 'o-')
plt.xlabel("Mode index")
plt.ylabel("Singular value (log scale)")
plt.title("Singular Value Decay")
plt.show()

# Obtain the reduced basis
V_r = V[:, :r]

# The solution is approximated as u ~ Vr*a(t)
# Therefore we can write the ode as d/dt(Vr*a) = A * (Vr*a)
# Projection onto the basis yield da/dt = Vr^T * A * Vr * a
# The reduced matrix A_red = Vr^T * A * Vr have the size r x r.
# We need the initial coefficients from the initial condition also
# which is a_0 = V_r^T * (u0 - mean)

A_red  = V_r.T @ A @ V_r
u0_red = V_r.T @ (u0)

# Integrate the reduced system
def rhs_reduced(t, a):
	return A_red@a

sol_red = solve_ivp(rhs_reduced, [t0, t1], u0_red, t_eval=t_eval, method='RK45')
A_sol = sol_red.y   # shape: (r,timesteps)

# Reconstruct the reduced state
U_rom = (V_r @ A_sol)


# Plotting
plt.figure(figsize=(12, 5))

# Plot a few states from both full and reduced
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(t_eval, U[i, :], 'k-', label='Full')
    plt.plot(t_eval, U_rom[i, :], 'r--', label='Reduced')
    plt.xlabel("Time")
    plt.ylabel(f"$u_{i}$")
    plt.title(f"State {i}")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

# Plot reconstruction error over time
error = np.linalg.norm(U - U_rom, axis=0)
plt.figure(figsize=(6,3))
plt.plot(t_eval, error)
plt.xlabel("Time")
plt.ylabel("Reconstruction Error (L2)")
plt.title("Full vs Reduced Order Solution Error")
plt.tight_layout()
plt.show()

