import matplotlib

matplotlib.use('TkAgg')

from pyproj import Proj
from shapely.affinity import rotate
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon


def latlon_to_utm(lat: list, lon: list):
    """
    Convert latitude and longitude to UTM coordinates.
    Returns:
    - x, y: lists of UTM x and y coordinates
    """
    zone_number = int((lon[0] + 180) / 6) + 1  # Determine UTM zone number
    p = Proj(proj='utm', zone=zone_number, ellps='WGS84')
    x, y = p(lon, lat)
    return x, y


def generate_random_hexagon():
    """Generate a random hexagon."""
    theta = np.linspace(0, 2 * np.pi, 7)
    x_hex = np.cos(theta)
    y_hex = np.sin(theta)
    perturb = np.random.uniform(-0.2, 0.2, size=(6, 2))
    x_hex[:-1] += perturb[:, 0]
    y_hex[:-1] += perturb[:, 1]
    x_hex[-1] = x_hex[0]
    y_hex[-1] = y_hex[0]
    return x_hex, y_hex


class LawnmowerPath:

    def __init__(self, lat_points, lon_points):
        self.lat_points = lat_points
        self.lon_points = lon_points
        self.polygon = Polygon(zip(self.lon_points, self.lat_points))

    def lawnmower_path_oriented(self, x_hex, y_hex, fov_width, orientation='horizontal'):
        """
         Generate lawnmower path for the given polygon.
        Parameters:
        - x_hex, y_hex: coordinates of the polygon vertices
        - fov_width: width of the drone's FOV
        - orientation: 'horizontal' or 'vertical'

        Returns:
        - list of path waypoints
        """
        polygon = Polygon(zip(x_hex, y_hex))
        if orientation == 'horizontal':
            start, end = polygon.bounds[0], polygon.bounds[2]
            lines = [LineString([(start, y), (end, y)]) for y in
                     np.arange(polygon.bounds[1], polygon.bounds[3], fov_width)]
        elif orientation == 'vertical':
            start, end = polygon.bounds[1], polygon.bounds[3]
            lines = [LineString([(x, start), (x, end)]) for x in
                     np.arange(polygon.bounds[0], polygon.bounds[2], fov_width)]
        else:
            raise ValueError("Orientation should be 'horizontal' or 'vertical'")
        path = []
        for i, line in enumerate(lines):
            intersection = polygon.intersection(line)
            if intersection.geom_type == 'MultiLineString':
                segments = list(intersection)
            else:
                segments = [intersection]
            for segment in segments:
                if i % 2 == 0:
                    path.extend(segment.coords)
                else:
                    path.extend(segment.coords[::-1])
        return path

    def plot_single_lawnmower_path(self, x_hex, y_hex, path, title):
        """Plot the hexagon and a single lawnmower path."""
        x_path, y_path = zip(*path)
        plt.figure(figsize=(8, 8))
        plt.fill(x_hex, y_hex, alpha=0.3, label='Hexagon')
        plt.plot(x_hex, y_hex, '-o', label='Hexagon Boundary')
        plt.plot(x_path, y_path, '-r', label='Coverage Path')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()

    def compute_rotated_path(self, x_hex, y_hex, fov_width, angle_degrees, orientation):
        """
        Compute the lawnmower path for a rotated polygon.

        Parameters:
        - x_hex, y_hex: coordinates of the polygon vertices
        - fov_width: width of the drone's FOV
        - angle_degrees: angle to rotate the polygon
        - orientation: 'horizontal' or 'vertical'

        Returns:
        - Rotated back path
        """
        polygon = Polygon(zip(x_hex, y_hex))
        rotated_polygon = rotate(polygon, angle_degrees, origin='centroid')
        x_rotated, y_rotated = rotated_polygon.exterior.xy
        rotated_path = self.lawnmower_path_oriented(x_rotated, y_rotated, fov_width, orientation)
        path_polygon = LineString(rotated_path)
        rotated_back_path_polygon = rotate(path_polygon, -angle_degrees, origin=rotated_polygon.centroid)
        return list(rotated_back_path_polygon.coords)

    def decompose_polygon(self):
        """Decompose the polygon into simple polygons."""
        lines = [LineString(self.polygon.exterior.coords)]
        merged_lines = unary_union(lines)
        return list(polygonize(merged_lines))

    def plot_decomposed_polygons(self, simple_polygons):
        """Plot the decomposed polygons."""
        plt.figure(figsize=(8, 8))
        for poly in simple_polygons:
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.3)
            plt.plot(x, y, '-o')
        plt.plot(self.lon_points, self.lat_points, '-o', label='Original Sequence')
        for i, (x, y) in enumerate(zip(self.lon_points, self.lat_points)):
            plt.annotate(i, (x, y), fontsize=10, ha='right')
        plt.title('Decomposed Simple Polygons with Original Sequence')
        plt.xlabel('Longitude (mock Cartesian X)')
        plt.ylabel('Latitude (mock Cartesian Y)')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()

    def compute_lawnmower_paths(self, simple_polygons, fov_width):
        """Compute the lawnmower paths for the decomposed polygons."""
        x1, y1 = simple_polygons[0].exterior.xy
        path1 = self.lawnmower_path_oriented(x1, y1, fov_width, orientation='horizontal')
        x2, y2 = simple_polygons[1].exterior.xy
        path2_reversed = self.lawnmower_path_oriented(x2, y2, fov_width, orientation='horizontal')[::-1]
        combined_path = path1 + path2_reversed
        return combined_path

    def plot_combined_lawnmower_path(self, simple_polygons, combined_path):
        """Plot the combined lawnmower path."""
        x_combined_path, y_combined_path = zip(*combined_path)
        plt.figure(figsize=(8, 8))
        x1, y1 = simple_polygons[0].exterior.xy
        x2, y2 = simple_polygons[1].exterior.xy
        plt.fill(x1, y1, alpha=0.3, label='Polygon 1')
        plt.plot(x1, y1, '-o', label='Polygon 1 Boundary')
        plt.fill(x2, y2, alpha=0.3, label='Polygon 2')
        plt.plot(x2, y2, '-o', label='Polygon 2 Boundary')
        plt.plot(x_combined_path, y_combined_path, '-r', label='Combined Coverage Path')
        plt.title('Simple Polygons with Combined Lawnmower Path')
        plt.xlabel('Longitude (mock Cartesian X)')
        plt.ylabel('Latitude (mock Cartesian Y)')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()

    def get_polygon_area(self):
        """Get the area of the polygon."""
        return self.polygon.area

    def test_multiple_angles(self, fov_width=0.1):
        x_hex, y_hex = generate_random_hexagon()

        paths_rotated = []
        lengths_rotated = []
        angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                           130, 140, 150, 160, 170])

        for angle in angles:
            for orientation in ['horizontal', 'vertical']:
                path = self.compute_rotated_path(x_hex, y_hex, fov_width, angle, orientation)
                paths_rotated.append((angle, orientation, path))
                length = sum(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for (x1, y1), (x2, y2) in zip(path, path[1:]))
                lengths_rotated.append(length)

        min_length_index = np.argmin(lengths_rotated)
        self.shortest_details = paths_rotated[min_length_index]
        self.shortest_length_rotated = lengths_rotated[min_length_index]

        max_length_index = np.argmax(lengths_rotated)
        self.longest_details = paths_rotated[max_length_index]
        self.longest_length_rotated = lengths_rotated[max_length_index]

        return x_hex, y_hex

    def visualize_best_worst_paths(self, x_hex, y_hex):
        self.plot_single_lawnmower_path(x_hex, y_hex, self.shortest_details[2],
                                        f'Shortest Lawnmower Path at {self.shortest_details[0]} Degrees ({self.shortest_details[1]} orientation)')
        self.plot_single_lawnmower_path(x_hex, y_hex, self.longest_details[2],
                                        f'Longest Lawnmower Path at {self.longest_details[0]} Degrees ({self.longest_details[1]} orientation)')


# Additional functionality can be added to this class as needed.
def runner():
    # Define the points
    lat_points = [40.7128, 40.7128, 40.7138, 40.7138, 40.7143, 40.7143, 40.7130, 40.7130, 40.7128]
    lon_points = [-74.0060, -74.0050, -74.0050, -74.0060, -74.0062, -74.0058, -74.0058, -74.0062, -74.0060]

    # Instantiate the LawnmowerPath class
    path_obj = LawnmowerPath(lat_points, lon_points)

    # Decompose the polygon
    simple_polygons = path_obj.decompose_polygon()

    # Plot decomposed polygons
    path_obj.plot_decomposed_polygons(simple_polygons)

    # Compute the lawnmower paths for the decomposed polygons
    fov_width = 0.0001
    combined_path = path_obj.compute_lawnmower_paths(simple_polygons, fov_width)

    # Plot combined lawnmower path
    path_obj.plot_combined_lawnmower_path(simple_polygons, combined_path)

    # Print the area of the polygon
    print(f"Area of the Polygon: {path_obj.get_polygon_area()} square meters")

    # Test for multiple angles and find the shortest path using the rotation approach
    x_hex, y_hex = path_obj.test_multiple_angles(fov_width=0.1)
    path_obj.visualize_best_worst_paths(x_hex, y_hex)


# Run the functions
runner()
