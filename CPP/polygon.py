import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.ops import cascaded_union


def generate_random_polygon(num_points=10):
    """
    Generate a random polygon with a given number of points.

    Parameters:
    - num_points (int): Number of points to use for the polygon.

    Returns:
    - polygon (Polygon): A Shapely Polygon object.
    """
    # Generate random points
    points = np.random.rand(num_points, 2) * 10

    # Use the Convex Hull of the points to ensure a simple polygon
    polygon = Polygon(points).convex_hull

    return polygon


def plot_polygon(polygon):
    """
    Plot the given polygon on a 2D plane.

    Parameters:
    - polygon (Polygon): A Shapely Polygon object.
    """
    plt.figure(figsize=(10, 7))
    x, y = polygon.exterior.xy
    plt.fill(x, y, alpha=0.5, fc='cyan', ec='black')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Randomly Generated Polygon')
    plt.grid(True)
    plt.show()


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    Parameters:
    - points (array-like): List of points.
    - alpha (float): Alpha parameter for the alpha shape. Lower values create a more concave shape.

    Returns:
    - polygon (Polygon): The resulting alpha shape as a Shapely Polygon object.
    """
    # Create the Delaunay triangulation of the points
    triangles = Delaunay(points)
    triangles = [Polygon(points[triangle]) for triangle in triangles.vertices]

    # Filter triangles based on the alpha parameter
    alpha_triangles = [
        triangle for triangle in triangles
        if triangle.area < alpha
    ]

    # Create the alpha shape by taking the union of the triangles
    alpha_shape = cascaded_union(alpha_triangles)

    return alpha_shape


def plot_shape(shape):
    """
    Plot the given shape (Polygon or MultiPolygon) on a 2D plane.

    Parameters:
    - shape (Polygon or MultiPolygon): A Shapely shape object.
    """
    plt.figure(figsize=(10, 7))

    if isinstance(shape, Polygon):
        x, y = shape.exterior.xy
        plt.fill(x, y, alpha=0.5, fc='cyan', ec='black')
    elif isinstance(shape, MultiPolygon):
        for polygon in shape.geoms:
            x, y = polygon.exterior.xy
            plt.fill(x, y, alpha=0.5, fc='cyan', ec='black')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Randomly Generated Shape')
    plt.grid(True)
    plt.show()


# Generate and plot a random polygon
polygon = generate_random_polygon()
plot_polygon(polygon)

# Generate random points
num_points = 80
points = np.random.rand(num_points, 2) * 10

# Compute the alpha shape
alpha = 1.5
concave_polygon = alpha_shape(points, alpha)

# Plot the resulting concave shape
plot_shape(concave_polygon)
