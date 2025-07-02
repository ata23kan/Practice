import numpy as np
import matplotlib.pyplot as plt

# Interpolate a grid using inverse distance weighting interpolation.
# - D. Shepard. A two-dimensional interpolation function for irregularly-spaced data, 
#   Proceedings of the 1968 ACM National Conference (1968) 517–524.


# Data points
pts = np.array([
	[0, 0, 10],
	[1, 0, 20],
	[0, 1, 30],
	[1, 1, 40]
	])

# Power parameter
c = 2

def idw(x, y, pts, c):

	x_data = pts[:, 0]
	y_data = pts[:, 1]
	z_data = pts[:, 2]

	# Calculate the distances from the given data
	r = np.sqrt((x-x_data)**2 + (y-y_data)**2)

	# Check if there are any zero distances
	if np.any(r==0):
		return z_data[r==0][0]

	# compute weights phi(r)
	phi = 1 / (r**c)

	# Compute weighted average
	return np.sum(phi*z_data) / np.sum(phi)

# Create a grid and interpolate the data
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

# Vectorize the IDW function for grid evaluation
idw_vectorized = np.vectorize(lambda x, y: idw(x, y, pts, c))
grid_z = idw_vectorized(grid_x, grid_y)

# Plot the interpolated surface
plt.figure(figsize=(8, 6))
contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='jet')
plt.colorbar(contour, label='Interpolated Z')
plt.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='jet', edgecolor='k', s=100, label='Data Points')
plt.title('Inverse Distance Weighting Interpolation')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.legend()
plt.show()
