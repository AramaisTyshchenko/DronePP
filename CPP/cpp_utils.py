"""
This module contains utility classes and methods for UAV path planning.
It includes methods for generating polygons, plotting, calculations, and path planning.
"""
import matplotlib

from CPP.helpers import calculate_angle_between_vectors, calculate_interior_angle

matplotlib.use('TkAgg')

import random
import numpy as np

import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.affinity import rotate
from shapely.ops import unary_union, polygonize

from scipy.spatial import ConvexHull, Delaunay


class PolygonGenerator:
    """
    Utility class containing methods for generating polygons and points.
    """

    @staticmethod
    def generate_random_lat_lon_points(num_points=9, lat_range=(40.7120, 40.7150), lon_range=(-74.0070, -74.0050)):
        """
          Generate a list of random latitude and longitude points.

          Parameters:
          - num_points (int): Number of random points to generate.
          - lat_range (tuple): Range for latitude.
          - lon_range (tuple): Range for longitude.

          Returns:
          - list: List of tuples containing latitude and longitude.
        """
        lat_points = [random.uniform(*lat_range) for _ in range(num_points)]
        lon_points = [random.uniform(*lon_range) for _ in range(num_points)]
        return lat_points, lon_points

    @staticmethod
    def generate_polygon_from_points(points):
        """
           Generate a convex polygon from a set of points using Convex Hull.

           Parameters:
           - points (list): List of tuples containing latitude and longitude points.

           Returns:
           - list: List of tuples representing the vertices of the convex polygon.
        """
        points_array = np.array(points)
        hull = ConvexHull(points_array)
        polygon_vertices = [tuple(points_array[i]) for i in hull.vertices]
        return polygon_vertices

    @staticmethod
    def generate_random_polygon(num_points=10):
        """
        Generate a random polygon with a given number of points.

        Parameters:
        - num_points (int): Number of points to use for the polygon.

        Returns:
        - polygon (Polygon): A Shapely Polygon object.
        """
        points = np.random.rand(num_points, 2) * 10
        polygon = Polygon(points).convex_hull
        return polygon


class PolygonUtils:
    """
    Utility class containing methods for calculations related to polygons.
    """

    @staticmethod
    def alpha_shape(points, alpha):
        """
        Compute the alpha shape (concave hull) for a given set of points.

        Parameters:
        - points: List of points to compute the alpha shape for.
        - alpha: Alpha value for the alpha shape algorithm.

        Returns:
        - A Polygon object representing the alpha shape.
        """
        # Create the Delaunay triangulation of the points
        triangles = Delaunay(points)
        triangles = [Polygon(points[triangle]) for triangle in triangles.vertices]
        alpha_triangles = [triangle for triangle in triangles if triangle.area < alpha]
        alpha_shape = unary_union(alpha_triangles)
        return alpha_shape

    @staticmethod
    def is_convex(polygon):
        """Check if a polygon is convex."""
        return polygon.is_valid and polygon.convex_hull.equals(polygon)

    @staticmethod
    def is_almost_convex(polygon, angle_threshold=180):
        coords = list(polygon.exterior.coords)
        n = len(coords) - 1  # The last point is the same as the first one
        for i in range(n):
            p1, p2, p3 = coords[i], coords[(i + 1) % n], coords[(i + 2) % n]
            interior_angle = calculate_interior_angle(p1, p2, p3)
            if interior_angle > angle_threshold:
                return False
        return True

    @staticmethod
    def is_ear(p1, p2, p3):
        """Check if angle formed by p1, p2, p3 is an ear."""
        # Compute vectors
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]

        # Cross product should be positive (indicating it's a convex vertex)
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        return cross_product > 0

    @staticmethod
    def ear_clipping(polygon):
        """Decompose a simple polygon into a list of convex polygons using ear clipping."""
        # Initialize list to store resulting convex polygons
        convex_polygons = []
        # Convert polygon into a list of vertices
        vertices = list(polygon.exterior.coords)[:-1]  # Exclude the last point because it's the same as the first

        while len(vertices) > 3:
            for i in range(len(vertices)):
                # Triplet of vertices
                p1 = vertices[i]
                p2 = vertices[(i + 1) % len(vertices)]
                p3 = vertices[(i + 2) % len(vertices)]
                # Create a candidate ear polygon
                ear_candidate = Polygon([p1, p2, p3])

                if PolygonUtils.is_ear(p1, p2, p3) and polygon.contains(ear_candidate):
                    # This is a valid ear. Cut it and add to the list of convex polygons.
                    convex_polygons.append(ear_candidate)
                    # Remove the ear tip from the list of vertices
                    del vertices[(i + 1) % len(vertices)]
                    # Stop the loop and re-run the algorithm
                    break

        # Add the remaining triangle
        convex_polygons.append(Polygon(vertices))
        return convex_polygons

    @staticmethod
    def hertel_mehlhorn(triangles, angle_threshold=180):
        merged_polygons = list(triangles)
        changes_made = True

        while changes_made:
            changes_made = False  # Reset flag
            i = 0
            while i < len(merged_polygons):
                polygon1 = merged_polygons[i]
                # Flag to break out of the inner loop
                merged_this_iteration = False
                j = i + 1
                while j < len(merged_polygons):
                    polygon2 = merged_polygons[j]
                    union = unary_union([polygon1, polygon2])

                    if union.geom_type == 'Polygon':
                        if PolygonUtils.is_almost_convex(union, angle_threshold):
                            merged_polygons[i] = union
                            del merged_polygons[j]

                            # Update flags
                            changes_made = True
                            merged_this_iteration = True
                            break
                        else:
                            j += 1
                    else:
                        j += 1
                # If we merged polygons, start over; otherwise, proceed
                if not merged_this_iteration:
                    i += 1
        return merged_polygons

    @staticmethod
    def decompose_into_convex(polygon):
        """Decompose a Polygon or MultiPolygon into a list of convex polygons."""
        if isinstance(polygon, Polygon):
            return PolygonUtils.ear_clipping(polygon)
        elif isinstance(polygon, MultiPolygon):
            convex_polygons = []
            for poly in polygon.geoms:
                convex_polygons.extend(PolygonUtils.ear_clipping(poly))
            return convex_polygons
        else:
            raise TypeError("Input must be a Polygon or MultiPolygon")

    @staticmethod
    def decompose_polygon(polygon, angle_threshold=180):
        """Decompose the polygon into simple polygons."""
        if isinstance(polygon, Polygon):
            # lines = [LineString(polygon.exterior.coords)]
            # merged_lines = unary_union(lines)
            # return list(polygonize(merged_lines))
            all_convex_polygons = []
            convex_triangles = PolygonUtils.decompose_into_convex(polygon)
            # Merge them into convex polygons
            convex_polygons = PolygonUtils.hertel_mehlhorn(convex_triangles, angle_threshold=angle_threshold)
            all_convex_polygons.extend(convex_polygons)
            return all_convex_polygons
        elif isinstance(polygon, MultiPolygon):
            all_simple_polygons = []
            for polygon in polygon.geoms:
                lines = [LineString(polygon.exterior.coords)]
                merged_lines = unary_union(lines)
                all_simple_polygons.extend(list(polygonize(merged_lines)))

            all_convex_polygons = []
            for simple_polygon in all_simple_polygons:
                convex_triangles = PolygonUtils.decompose_into_convex(simple_polygon)
                # Merge them into convex polygons
                convex_polygons = PolygonUtils.hertel_mehlhorn(convex_triangles, angle_threshold=angle_threshold)
                all_convex_polygons.extend(convex_polygons)

            return all_convex_polygons
        else:
            raise TypeError("The polygon attribute must be a Polygon or MultiPolygon")


class PathPlannerUtils:
    """
    Utility class containing methods for path planning.
    """

    @staticmethod
    def calculate_path_length(path):
        """
           Calculate the length of a given path.

           Parameters:
           - path: list of waypoints

           Returns:
           - Total length of the path
        """
        length = 0
        for i in range(1, len(path)):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return length

    @staticmethod
    def count_turns_in_path(path, angle_threshold=30):
        """Count the number of turns in a path based on angle changes."""
        num_turns = 0
        for i in range(len(path) - 2):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            x3, y3 = path[i + 2]
            v1 = (x2 - x1, y2 - y1)
            v2 = (x3 - x2, y3 - y2)
            angle = calculate_angle_between_vectors(v1, v2)
            if abs(angle) > angle_threshold:
                num_turns += 1
        return num_turns

    @staticmethod
    def lawnmower_path(polygon, fov_width, orientation='horizontal'):
        """
         Generate lawnmower path for the given polygon.
        Parameters:
        - polygon
        - fov_width: width of the drone's FOV
        - orientation: 'horizontal' or 'vertical'

        Returns:
        - list of path waypoints
        """
        # x_hex, y_hex: coordinates of the polygon vertices
        x_hex, y_hex = polygon.exterior.xy
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
                segments = list(intersection.geoms)
            else:
                segments = [intersection]
            for segment in segments:
                if i % 2 == 0:
                    path.extend(segment.coords)
                else:
                    path.extend(segment.coords[::-1])
        return path

    @staticmethod
    def generate_single_spiral(poly, fov_width, steps):
        spiral_path = []
        tmp_poly = poly
        for _ in range(steps):
            # Check if tmp_poly has become a MultiPolygon
            if tmp_poly.geom_type == 'MultiPolygon':
                for sub_poly in tmp_poly.geoms:
                    x, y = sub_poly.exterior.coords.xy
                    spiral_path.extend(list(zip(x, y)))
                break  # Stop the process for this polygon

            # Get the exterior coordinates of the shrunken polygon
            x, y = tmp_poly.exterior.coords.xy
            spiral_path.extend(list(zip(x, y)))

            # Shrink the polygon
            tmp_poly = tmp_poly.buffer(-fov_width, join_style=3)

            # Break if polygon has vanished or become a line
            if tmp_poly.is_empty or tmp_poly.geom_type not in ["Polygon", "MultiPolygon"]:
                break
        return spiral_path

    @staticmethod
    def compute_rotated_path(polygon, fov_width, angle_degrees=0):
        """
        Compute the lawnmower path for a rotated polygon.

        Parameters:
        - fov_width: width of the drone's FOV
        - angle_degrees: angle to rotate the polygon

        Returns:
        - Rotated back path
        """
        rotated_polygon = rotate(polygon, angle_degrees, origin='centroid')
        rotated_path = PathPlannerUtils.lawnmower_path(rotated_polygon, fov_width)

        path_polygon = LineString(rotated_path)
        rotated_back_path_polygon = rotate(path_polygon, -angle_degrees, origin=rotated_polygon.centroid)
        return list(rotated_back_path_polygon.coords)

    @staticmethod
    def find_best_path_for_polygon(polygon, fov_width, angles):
        best_path = None
        best_length = float('inf')

        for angle in angles:
            path = PathPlannerUtils.compute_rotated_path(polygon, fov_width, angle)
            length = PathPlannerUtils.calculate_path_length(path)

            if length < best_length:
                best_length = length
                best_path = path

        return best_path, best_length


class PlottingUtils:
    """Utility class containing methods for plotting."""
    colors = [
        'blue', 'orange', 'green', 'red', 'purple',
        'brown', 'pink', 'grey', 'yellow', 'cyan',
        'magenta', 'lime', 'gold', 'silver', 'darkblue',
        'darkcyan', 'darkmagenta', 'darkorange', 'darkgreen', 'darkred',
        'darkgrey', 'darkviolet', 'darkbrown', 'darkyellow'
    ]

    @staticmethod
    def _setup_plot(title, x_label='X Coordinate', y_label='Y Coordinate'):
        plt.figure(figsize=(15, 8))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)

    @staticmethod
    def _plot_polygon(shape, color='cyan'):
        """
        :param shape: The shape to be plotted
        :param color: The fill color for the polygon
        :return:
        """
        if isinstance(shape, Polygon):
            x, y = shape.exterior.xy
            plt.fill(x, y, alpha=0.5, fc=color, ec='black')
        elif isinstance(shape, MultiPolygon):
            for polygon in shape.geoms:
                x, y = polygon.exterior.xy
                plt.fill(x, y, alpha=0.5, fc=color, ec='black')

    @staticmethod
    def _plot_points(points):
        """
        PlottingUtils.plot_elements([points, polygon], title='Random Points and Convex Hull')

        :param points:
        :return:
        """
        lats, lons = zip(*points)
        plt.scatter(lons, lats, c='blue', label='Random Points')

    @staticmethod
    def _plot_delaunay(tri):
        """
        PlottingUtils.plot_elements([Delaunay(np.array(points))], title='Delaunay Triangulation')

        :param tri:
        :return:
        """
        plt.triplot(tri.points[:, 1], tri.points[:, 0], tri.simplices.copy())
        plt.scatter(tri.points[:, 1], tri.points[:, 0], c='blue', label='Random Points')

    @staticmethod
    def _mark_start_end_points(paths):
        plt.scatter(paths[0][0][0], paths[0][0][1], marker='s', color='green', zorder=5, s=100)
        plt.scatter(paths[-1][-1][0], paths[-1][-1][1], marker='s', color='red', zorder=5, s=100)

    @staticmethod
    def _plot_common_elements(paths, simple_polygons, title):
        """Helper method for common plotting operations."""
        PlottingUtils._setup_plot(title)
        previous_end_point = None
        for i, (poly, path) in enumerate(zip(simple_polygons, paths)):
            current_color = PlottingUtils.colors[i % len(PlottingUtils.colors)]

            # Plot the polygon with its edges
            PlottingUtils._plot_polygon(poly, color='white')
            x, y = poly.exterior.xy
            plt.scatter(x, y, marker='o', color=current_color, zorder=1)

            # If there's a previous ending point, connect it to the current starting point
            if previous_end_point:
                plt.plot([previous_end_point[0], path[0][0]], [previous_end_point[1], path[0][1]], '--',
                         color=current_color, zorder=2)

                # Calculate the midpoint for the arrow
                mid_x = (previous_end_point[0] + path[0][0]) / 2
                mid_y = (previous_end_point[1] + path[0][1]) / 2
                dx = (path[0][0] - mid_x) / 2
                dy = (path[0][1] - mid_y) / 2
                plt.arrow(mid_x, mid_y, dx, dy, shape='full', lw=0, length_includes_head=True, head_width=.25,
                          color=current_color, zorder=3)

            # Plot the path
            x_path, y_path = zip(*path)
            plt.plot(x_path, y_path, color=current_color)
            previous_end_point = path[-1]

        PlottingUtils._mark_start_end_points(paths)
        plt.axis('equal')

    @staticmethod
    def plot_elements(elements, title, x_label='X Coordinate', y_label='Y Coordinate', show=True):
        PlottingUtils._setup_plot(title, x_label, y_label)

        type_to_function = {
            Polygon: PlottingUtils._plot_polygon,
            MultiPolygon: PlottingUtils._plot_polygon,
            Delaunay: PlottingUtils._plot_delaunay,
            list: PlottingUtils._plot_points  # assuming list of points
        }

        for element in elements:
            plot_func = type_to_function.get(type(element), lambda x: print(f"Unsupported element type: {type(x)}"))
            plot_func(element)

        if show:
            plt.show()

    @staticmethod
    def plot_decomposed_polygons(simple_polygons):
        title = 'Decomposed Simple Polygons with Original Sequence'
        x_label = 'Longitude (mock Cartesian X)'
        y_label = 'Latitude (mock Cartesian Y)'

        PlottingUtils._setup_plot(title, x_label, y_label)

        for i, poly in enumerate(simple_polygons):
            current_color = PlottingUtils.colors[i % len(PlottingUtils.colors)]
            PlottingUtils._plot_polygon(poly, color=current_color)
            x, y = poly.exterior.xy
            plt.plot(x, y, '-o', color=current_color)

        plt.legend()
        plt.axis('equal')
        plt.show()

    @staticmethod
    def plot_combined_lawnmower_path(simple_polygons, combined_path, title):
        PlottingUtils._plot_common_elements(combined_path, simple_polygons, title)
        plt.show()

    @staticmethod
    def _create_title(description, details):
        return (
            f"{description} Lawnmower Path at {details['details'][0]} Degrees \n"
            f"Path Length: {details['length']:.2f} units \n"
            f"Total Turns: {details['turns']} \n"
            f"Total Transition Distance: {details['transition_distance']:.2f} units"
        )

    @staticmethod
    def visualize_best_worst_paths(simple_polygons, shortest_details, longest_details):
        shortest_title = PlottingUtils._create_title("Shortest", shortest_details)
        longest_title = PlottingUtils._create_title("Longest", longest_details)

        PlottingUtils.plot_combined_lawnmower_path(simple_polygons, shortest_details['details'][1],
                                                   title=shortest_title)
        PlottingUtils.plot_combined_lawnmower_path(simple_polygons, longest_details['details'][1],
                                                   title=longest_title)
