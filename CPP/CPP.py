import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


def generate_random_points(num_points=50, lat_range=(50.0, 51.0), lon_range=(-1.0, 0.0)):
    """
    Generate a list of random latitude and longitude points.

    Parameters:
    - num_points (int): Number of random points to generate.
    - lat_range (tuple): Range for generating random latitude values.
    - lon_range (tuple): Range for generating random longitude values.

    Returns:
    - list: List of tuples containing latitude and longitude points.
    """
    points = [(random.uniform(*lat_range), random.uniform(*lon_range)) for _ in range(num_points)]
    return points


def generate_polygon_from_points(points):
    """
    Generate a convex polygon from a set of points using Convex Hull.

    Parameters:
    - points (list): List of tuples containing latitude and longitude points.

    Returns:
    - list: List of tuples representing the vertices of the convex polygon.
    """
    # Convert list of tuples to numpy array
    points_array = np.array(points)

    # Calculate the convex hull
    hull = ConvexHull(points_array)

    # Extract the vertices of the convex polygon
    polygon_vertices = [tuple(points_array[i]) for i in hull.vertices]

    return polygon_vertices


def plot_points_and_polygon(points, polygon):
    """
    Plot the given points and polygon on a 2D plane.

    Parameters:
    - points (list): List of tuples containing latitude and longitude points.
    - polygon (list): List of tuples representing the vertices of the convex polygon.
    """
    # Extract latitude and longitude for plotting
    lats, lons = zip(*points)
    poly_lats, poly_lons = zip(*polygon)

    plt.figure(figsize=(10, 7))
    plt.scatter(lons, lats, c='blue', label='Random Points')
    plt.plot(poly_lons, poly_lats, c='red', label='Convex Hull')
    plt.fill(poly_lons, poly_lats, alpha=0.2, color='red')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Random Points and Convex Hull')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_delaunay_triangulation(points):
    """
    Plot the given points and their Delaunay triangulation on a 2D plane.

    Parameters:
    - points (list): List of tuples containing latitude and longitude points.
    """
    # Convert list of tuples to numpy array for triangulation
    points_array = np.array(points)

    # Compute Delaunay triangulation
    tri = Delaunay(points_array)

    plt.figure(figsize=(10, 7))
    plt.triplot(points_array[:, 1], points_array[:, 0], tri.simplices.copy())
    plt.scatter(points_array[:, 1], points_array[:, 0], c='blue', label='Random Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Delaunay Triangulation')
    plt.legend()
    plt.grid(True)
    plt.show()


# Generate 50 random points
random_points = generate_random_points()
random_points[:10]  # Display the first 10 points

# Generate a convex polygon from the random points
polygon_vertices = generate_polygon_from_points(random_points)

# Plot the random points and the convex polygon
plot_points_and_polygon(random_points, polygon_vertices)

# Define the Area:
#
# Define the area to be covered in terms of its boundaries (latitude, longitude, altitude). This can be represented as a polygon.
# Break down the area into a grid or set of waypoints. The size of each grid cell can be based on the drone's camera field of view and desired altitude.
# Choose a Strategy:
#
# Boustrophedon (Lawnmower) Pattern: This is one of the simplest paths where the drone covers the area in a back-and-forth manner, similar to mowing a lawn.
# Spiral Pattern: The drone starts from the outside and spirals inward or vice versa.
# Zig-Zag Pattern: A variation of the lawnmower pattern but with diagonal movements.
# Spanning Tree: Create a spanning tree of the area and traverse it. Useful for irregular areas.
# Advanced algorithms can consider obstacles, no-fly zones, etc.
