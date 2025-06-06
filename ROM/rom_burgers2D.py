import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# --- Parameters ---
nx, ny = 80, 80
Lx, Ly = 1.0, 1.0
dx, dy = Lx / nx, Ly / ny
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')
nu = 0.01
timesteps = 100
Tmax = 2.0
t_eval = np.linspace(0, Tmax, timesteps)

# --- Initial Condition: Gaussian ---
u0 = np.exp(-30 * ((X - Lx/2)**2 + (Y - Ly/2)**2))

# --- Finite Difference RHS ---
def rhs(t, uvec):
    u = uvec.reshape((nx, ny))
    # Periodic BCs using np.roll
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    d2udx2 = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
    d2udy2 = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2
    rhs = -u * dudx - u * dudy + nu * (d2udx2 + d2udy2)
    return rhs.ravel()

# --- Solve Full System ---
sol = solve_ivp(rhs, [0, Tmax], u0.ravel(), t_eval=t_eval, method='RK45')
U = sol.y  # shape: (nx*ny, timesteps)

# --- POD with Mean Subtraction ---
U_mean = U.mean(axis=1, keepdims=True)
U_centered = U - U_mean
V, S, WT = np.linalg.svd(U_centered, full_matrices=False)

# Singular value decay
plt.figure()
plt.semilogy(S, 'o-')
plt.xlabel("Mode index")
plt.ylabel("Singular value")
plt.title("Singular Value Decay")
plt.show()

# Select number of energetic modes
energy = np.cumsum(S**2) / np.sum(S**2)
r = np.searchsorted(energy, 0.97) + 1
print(f"Using r = {r} modes to capture 97% of the energy.")

# r = 5

V_r = V[:, :r]
u0_red = V_r.T @ (u0.ravel() - U_mean.squeeze())

# --- ROM Galerkin system ---
def rhs_rom(t, a):
    u_rec = V_r @ a + U_mean.squeeze()
    u = u_rec.reshape((nx, ny))
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    d2udx2 = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
    d2udy2 = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2
    residual = -u * dudx - u * dudy + nu * (d2udx2 + d2udy2)
    return V_r.T @ residual.ravel()

sol_red = solve_ivp(rhs_rom, [0, Tmax], u0_red, t_eval=t_eval, method='RK45')
A_sol = sol_red.y  # (r, timesteps)
U_rom = V_r @ A_sol + U_mean  # (nx*ny, timesteps)

# --- Animation ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(U[:,0].reshape(nx,ny), origin='lower', cmap='jet', vmin=U.min(), vmax=U.max())
axes[0].set_title('Full Solution')
cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.set_label('u')

im1 = axes[1].imshow(U_rom[:,0].reshape(nx,ny), origin='lower', cmap='jet', vmin=U.min(), vmax=U.max())
axes[1].set_title('ROM Solution')
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_label('u (ROM)')

im2 = axes[2].imshow((U[:,0] - U_rom[:,0]).reshape(nx,ny), origin='lower', cmap='RdBu_r')
axes[2].set_title('Pointwise Error')
cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.set_label('Error')

suptitle = fig.suptitle(f"Time = {t_eval[0]:.3f}", fontsize=16)

def update(frame):
    # Update solution images
    im0.set_data(U[:,frame].reshape(nx,ny))
    im1.set_data(U_rom[:,frame].reshape(nx,ny))
    # Update error image and adaptive colorbar
    err_frame = (U[:,frame] - U_rom[:,frame]).reshape(nx,ny)
    im2.set_data(err_frame)
    vmax = np.max(np.abs(err_frame))
    im2.set_clim(-vmax, vmax)
    cbar2.set_ticks([-vmax, 0, vmax])
    cbar2.set_ticklabels([f'{-vmax:.2e}', '0', f'{vmax:.2e}'])
    suptitle.set_text(f"Time = {t_eval[frame]:.3f}")
    return im0, im1, im2

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=False)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()