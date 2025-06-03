import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Grid and problem parameters
nx, ny = 50, 50
N = nx * ny
Lx, Ly = 1.0, 1.0
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
D = 0.1  # Diffusivity
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



# Pick time to visualize
t_index = timesteps//2
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.title("Full order")
plt.imshow(U[:, t_index].reshape((nx, ny)), origin='lower', cmap='jet')
plt.colorbar(label="u", fraction=0.046, pad=0.04)

plt.subplot(1,3,2)
plt.title("Reduced Order")
plt.imshow(U_rom[:, t_index].reshape((nx, ny)), origin='lower', cmap='jet')
plt.colorbar(label="u (ROM)", fraction=0.046, pad=0.04)

plt.subplot(1, 3, 3)
error_field = U[:, t_index] - U_rom[:, t_index]
plt.title("Pointwise Error (Full - Reduced)")
plt.imshow(error_field.reshape((nx, ny)), origin='lower', cmap='jet')
plt.colorbar(label="Error", fraction=0.046, pad=0.04)
plt.xlabel("x")
plt.ylabel("y")

plt.suptitle(f"Solution at time = {t_eval[t_index]:.3f}", fontsize=12)

plt.tight_layout()
plt.show()

# Error over time
error = np.linalg.norm(U - U_rom, axis=0)
plt.figure()
plt.plot(t_eval, error)
plt.xlabel("Time")
plt.ylabel("L2 Error")
plt.title("Reconstruction Error vs Time")
plt.show()
