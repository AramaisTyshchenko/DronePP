from itertools import product

import numpy as np
from pyproj import Proj
from scipy.spatial.distance import euclidean


def latlon_to_utm(lat: list, lon: list) -> tuple:
    """
    Convert latitude and longitude to UTM coordinates.

    Parameters:
    - lat (list): List of latitude coordinates.
    - lon (list): List of longitude coordinates.

    Returns:
    - tuple: Lists of UTM x and y coordinates.
    """
    zone_number = int((lon[0] + 180) / 6) + 1
    p = Proj(proj='utm', zone=zone_number, ellps='WGS84')
    x, y = p(lon, lat)
    return x, y


def calculate_angle_between_vectors(v1, v2):
    """
     Calculate the angle in degrees between vectors 'v1' and 'v2'.

     Parameters:
     - v1, v2: The vectors for which to calculate the angle, each as a tuple of (x, y).

     Returns:
     - angle in degrees
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calculate_interior_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)
    angle_rad = np.arctan2(np.linalg.norm(cross_product), dot_product)
    angle_deg = np.degrees(angle_rad)

    # The angle is concave if the cross product is negative.
    if cross_product < 0:
        angle_deg = 360 - angle_deg
    return angle_deg


def find_closest_point(target_point, points):
    """Find the closest point to a given target_point in a list of points."""
    min_distance = float('inf')
    closest_point = None
    for point in points:
        distance = euclidean(target_point, point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point


def is_line_in_polygon(line, polygons):
    for polygon in polygons:
        if polygon.contains(line):
            return True
    return False


def calculate_transition_distances(paths):
    """
    Calculate the shortest transition distance between each pair of polygons.

    Parameters:
    - paths: List of lawnmower paths for each polygon.

    Returns:
    - Dictionary with keys as (i, j) representing indices of the two polygons,
      and values as the shortest distance between them.
    """
    transition_distances = {}
    for i, j in product(range(len(paths)), repeat=2):
        if i == j:
            continue  # Skip same polygon

        end_points_i = paths[i][-1]  # Last point in path i
        start_points_j = paths[j][0]  # First point in path j

        # Calculate Euclidean distance between the end points of path i and start points of path j
        distance = np.sqrt((end_points_i[0] - start_points_j[0]) ** 2 + (end_points_i[1] - start_points_j[1]) ** 2)

        # Update the transition distance between polygons i and j
        transition_distances[(i, j)] = distance

    return transition_distances


def optimize_sequence_nearest_neighbor(lawnmower_paths, transition_distances):
    """
    Optimize the sequence of polygons to visit using the nearest-neighbor algorithm.

    Parameters:
    - transition_distances: Dictionary containing the distances between each pair of polygons.

    Returns:
    - List of indices representing the optimized sequence of polygons to visit.
    """
    # Start with the first polygon as the initial point
    current_polygon = 0
    remaining_polygons = set(range(1, len(lawnmower_paths)))

    optimized_sequence = [current_polygon]

    while remaining_polygons:
        # Find the nearest neighboring polygon
        nearest_neighbor = min(
            remaining_polygons,
            key=lambda j: transition_distances.get((current_polygon, j), float('inf'))
        )

        # Move to the nearest neighboring polygon
        optimized_sequence.append(nearest_neighbor)
        current_polygon = nearest_neighbor
        remaining_polygons.remove(nearest_neighbor)

    return optimized_sequence
