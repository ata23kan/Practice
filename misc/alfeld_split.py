import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def selective_alfeld_split(nodes, triangles, detect_func):
    """
    Apply selective Alfeld split to a 2D triangular mesh.

    Parameters:
    nodes : ndarray
        Nx2 array of node coordinates.
    triangles : ndarray
        Mx3 array of triangle vertex indices.
    detect_func : function
        Function that takes the vertices of a triangle and returns True if the triangle should be split.

    Returns:
    new_nodes : ndarray
        Updated array of node coordinates.
    new_triangles : ndarray
        Updated array of triangle connectivity.
    split_flags : list
        Flags indicating whether each triangle was split (for coloring).
    """

    new_nodes = nodes.copy()
    new_triangles = []
    split_flags = []  # This will be a simple flag array

    for tri in triangles:
        v1, v2, v3 = tri
        p1 = nodes[v1]
        p2 = nodes[v2]
        p3 = nodes[v3]

        # Check if the current triangle should be split
        if detect_func(np.array([p1, p2, p3])):
            centroid = (p1 + p2 + p3) / 3
            centroid_index = len(new_nodes)
            new_nodes = np.vstack([new_nodes, centroid])

            # Create new triangles with centroid
            new_triangles.append([v1, v2, centroid_index])
            new_triangles.append([v2, v3, centroid_index])
            new_triangles.append([v3, v1, centroid_index])
            
            # Flag these triangles as split (1)
            split_flags.extend([1, 1, 1])
        else:
            new_triangles.append([v1, v2, v3])
            
            # Flag this triangle as not split (0)
            split_flags.append(0)

    return np.array(new_nodes), np.array(new_triangles), np.array(split_flags)

def plot_mesh(nodes, triangles, split_flags):
    """
    Plot the mesh with the given node coordinates and triangle connectivity.
    """
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')

    triangulation = Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    plt.tripcolor(triangulation, split_flags, edgecolors='k', cmap='coolwarm')

    plt.title('Selective Alfeld Split')
    plt.show()

# Example usage

# Example mesh data
nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
triangles = np.array([[0, 1, 2], [1, 3, 2]])

# Detection function: Split triangles with area greater than 0.4
detect_func = lambda verts: 0.5 * np.abs(np.linalg.det(np.array([verts[1] - verts[0], verts[2] - verts[0]]))) > 0.4

# Apply Selective Alfeld Split
new_nodes, new_triangles, split_flags = selective_alfeld_split(nodes, triangles, detect_func)

# Plot the result
plot_mesh(new_nodes, new_triangles, split_flags)
