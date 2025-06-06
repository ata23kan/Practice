import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid and problem parameters
nx, ny = 50, 50
N = nx * ny
Lx, Ly = 1.0, 1.0
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
D = 0.01  # Diffusivity
timesteps = 100
Tmax = 1.0
t_eval = np.linspace(0, Tmax, timesteps)
r = 10  # POD modes

# Helper for 2D Laplacian with Dirichlet BCs
def laplacian_2d(u):
    u = u.reshape((nx, ny))
    lap = np.zeros_like(u)
    # 2D central difference for interior points
    lap[1:-1,1:-1] = (
        (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2 +
        (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dy**2
    )
    return lap.ravel()

# Initial condition: Gaussian bump at center
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), indexing='ij')
u0 = np.exp(-50 * ((X - Lx/2)**2 + (Y - Ly/2)**2)).ravel()

def rhs(t, u):
    return D * laplacian_2d(u)

sol = solve_ivp(rhs, [0, Tmax], u0, t_eval=t_eval, method='RK45')
U = sol.y  # Shape: (N, timesteps)


# SVD for POD basis (no mean subtraction)
V, S, WT = np.linalg.svd(U, full_matrices=False)
V_r = V[:, :r]


# Singular value decay
plt.figure()
plt.semilogy(S, 'o-')
plt.xlabel("Mode index")
plt.ylabel("Singular value")
plt.title("Singular Value Decay")
plt.show()


# Project A (matrix-free: use laplacian in reduced basis)
def apply_A(v):
    return D * laplacian_2d(v)

A_red = np.zeros((r, r))
for i in range(r):
    Av = apply_A(V_r[:, i])
    for j in range(r):
        A_red[j, i] = np.dot(V_r[:, j], Av)

u0_red = V_r.T @ u0

def rhs_reduced(t, a):
    return A_red @ a

sol_red = solve_ivp(rhs_reduced, [0, Tmax], u0_red, t_eval=t_eval, method='RK45')
A_sol = sol_red.y

U_rom = V_r @ A_sol  # Shape: (N, timesteps)



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Set up images for the three subplots
im0 = axes[0].imshow(U[:, 0].reshape((nx, ny)), origin='lower', cmap='jet', vmin=U.min(), vmax=U.max())
axes[0].set_title("Full order")
cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.set_label("u")

im1 = axes[1].imshow(U_rom[:, 0].reshape((nx, ny)), origin='lower', cmap='jet', vmin=U.min(), vmax=U.max())
axes[1].set_title("Reduced Order")
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_label("u (ROM)")

im2 = axes[2].imshow((U[:, 0] - U_rom[:, 0]).reshape((nx, ny)), origin='lower', cmap='jet')
axes[2].set_title("Pointwise Error (Full - Reduced)")
cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.set_label("Error")

suptitle = fig.suptitle(f"Solution at time = {t_eval[0]:.3f}", fontsize=14)

def update(frame):
    im0.set_data(U[:, frame].reshape((nx, ny)))
    im1.set_data(U_rom[:, frame].reshape((nx, ny)))
    err_frame = (U[:, frame] - U_rom[:, frame]).reshape((nx, ny))
    im2.set_data(err_frame)
    vmax = np.max(np.abs(err_frame))
    im2.set_clim(-vmax, vmax)
    cbar2.set_ticks([-vmax, 0, vmax])
    cbar2.set_ticklabels([f'{-vmax:.2e}', '0', f'{vmax:.2e}'])
    suptitle.set_text(f"Solution at time = {t_eval[frame]:.3f}")
    return im0, im1, im2

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=40, blit=False)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# Error over time
error = np.linalg.norm(U - U_rom, axis=0)
plt.figure()
plt.plot(t_eval, error)
plt.xlabel("Time")
plt.ylabel("L2 Error")
plt.title("Reconstruction Error vs Time")
plt.show()
