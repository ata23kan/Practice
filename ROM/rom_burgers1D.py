import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
nx = 100         # Number of spatial points
L = 1.0
x = np.linspace(0, L, nx)
dx = x[1] - x[0]
nu = 0.01        # Viscosity
timesteps = 200
Tmax = 2.0
t_eval = np.linspace(0, Tmax, timesteps)

# Initial condition: sine wave
u0 = np.sin(2 * np.pi * x)

# Periodic boundary with ghost nodes 
def periodic(u):
    u[0] = u[-2]
    u[-1] = u[1]
    return u

# Burgers' RHS using central differences
def rhs(t, u):
    u = u.copy()
    u = periodic(u)
    dudx = (u[2:] - u[:-2]) / (2 * dx)
    d2udx2 = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    out = -u[1:-1] * dudx + nu * d2udx2
    # Return array of size nx
    result = np.zeros_like(u)
    result[1:-1] = out
    # Enforce periodic BCs
    result[0] = result[-2]
    result[-1] = result[1]
    return result

# Collect snapshots
sol = solve_ivp(rhs, [0, Tmax], u0, t_eval=t_eval, method='RK45', vectorized=False)
U = sol.y   # shape (nx, timesteps)

# --- MEAN SUBTRACTION ---
U_mean = np.mean(U, axis=1, keepdims=True)
U_centered = U - U_mean

# --- POD ---
V, S, WT = np.linalg.svd(U_centered, full_matrices=False)
energy = np.cumsum(S**2) / np.sum(S**2)
# r = np.searchsorted(energy, 0.97) + 1
# print(f"Using r = {r} modes to capture 97% of the energy.")
r = 5

V_r = V[:, :r]

# Singular value decay
plt.figure()
plt.semilogy(S, 'o-')
plt.xlabel("Mode index")
plt.ylabel("Singular value")
plt.title("Singular Value Decay")
plt.show()

# --- Galerkin ROM ---

# Project initial condition (centered!)
u0_red = V_r.T @ (u0 - U_mean.squeeze())

# Build reduced nonlinear RHS (Galerkin)
def rhs_rom(t, a):
    # Recover field in physical space (centered)
    u_rec = V_r @ a + U_mean.squeeze()
    u_rec = periodic(u_rec.copy())
    dudx = (u_rec[2:] - u_rec[:-2]) / (2 * dx)
    d2udx2 = (u_rec[2:] - 2 * u_rec[1:-1] + u_rec[:-2]) / dx**2
    # Galerkin projection: compute the residual, then project
    residual = np.zeros_like(u_rec)
    residual[1:-1] = -u_rec[1:-1] * dudx + nu * d2udx2
    # Project residual onto POD modes
    return V_r.T @ residual

# Integrate ROM
sol_red = solve_ivp(rhs_rom, [0, Tmax], u0_red, t_eval=t_eval, method='RK45')
A_sol = sol_red.y  # shape (r, timesteps)
U_rom = V_r @ A_sol + U_mean  # reconstruct physical field

# --- Visualization ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
line_full, = ax1.plot(x, U[:, 0], 'k-', label='Full')
line_rom, = ax1.plot(x, U_rom[:, 0], 'r--', label='ROM')
ax1.set_ylim(np.min(U), np.max(U))
ax1.set_xlabel("x")
ax1.set_ylabel("u")
ax1.legend()
ax1.set_title("Solution and ROM")

line_err, = ax2.plot(x, U[:, 0] - U_rom[:, 0], 'b-')
ax2.set_ylim(-np.max(np.abs(U - U_rom)), np.max(np.abs(U - U_rom)))
ax2.set_xlabel("x")
ax2.set_ylabel("Pointwise Error")
ax2.set_title("Pointwise Error (Full - ROM)")

suptitle = fig.suptitle(f"Time = {t_eval[0]:.3f}", fontsize=16)

def update(frame):
    line_full.set_ydata(U[:, frame])
    line_rom.set_ydata(U_rom[:, frame])
    line_err.set_ydata(U[:, frame] - U_rom[:, frame])
    suptitle.set_text(f"Time = {t_eval[frame]:.3f}")
    return line_full, line_rom, line_err, suptitle

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=40, blit=False)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
