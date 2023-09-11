from typing import Union

import matplotlib
import numpy as np

from CPP.helpers import calculate_transition_distances, optimize_sequence_nearest_neighbor

matplotlib.use('TkAgg')

from shapely.geometry import Polygon, MultiPolygon
from cpp_utils import PolygonUtils, PathPlannerUtils, PlottingUtils

from typing import List, Tuple


def calculate_metrics(paths: List[List[Tuple[float, float]]],
                      optimized_sequence: List[int],
                      transition_distances: dict) -> Tuple[float, float, float]:
    """
    Calculate various metrics for a given set of paths.

    Parameters:
    - paths: A list of paths, where each path is a list of (x, y) coordinates.
    - optimized_sequence: A list of indices representing the optimized sequence of paths.
    - transition_distances: A dictionary representing the transition distances between paths.

    Returns:
    - total_length: The total length of the optimized paths.
    - total_turns: The total number of turns in the optimized paths.
    - total_transition_distance: The total transition distance between the optimized paths.
    """
    total_length, total_turns, total_transition_distance = 0, 0, 0

    for i, index in enumerate(optimized_sequence):
        path = paths[index]
        total_length += PathPlannerUtils.calculate_path_length(path)
        total_turns += PathPlannerUtils.count_turns_in_path(path)

        if i < len(optimized_sequence) - 1:
            next_index = optimized_sequence[i + 1]
            total_transition_distance += transition_distances.get((index, next_index), 0)

    return total_length, total_turns, total_transition_distance


def calculate_spiral_metrics(spirals):
    total_path_length = 0
    total_turns = 0
    transition_distances = calculate_transition_distances(spirals)
    optimized_sequence = optimize_sequence_nearest_neighbor(spirals, transition_distances)
    transition_distance = 0

    for i, idx in enumerate(optimized_sequence):
        path = spirals[idx]
        total_path_length += PathPlannerUtils.calculate_path_length(path)
        total_turns += PathPlannerUtils.count_turns_in_path(path)

        if i < len(optimized_sequence) - 1:
            next_idx = optimized_sequence[i + 1]
            transition_distance += transition_distances.get((idx, next_idx), 0)

    return total_path_length, total_turns, transition_distance, optimized_sequence


def calculate_total_transition_distance(paths):
    total_transition_distance = 0
    transition_distances = calculate_transition_distances(paths)
    optimized_sequence = optimize_sequence_nearest_neighbor(paths, transition_distances)

    for i in range(len(optimized_sequence) - 1):
        total_transition_distance += transition_distances.get((optimized_sequence[i], optimized_sequence[i + 1]), 0)

    return total_transition_distance, optimized_sequence


class CPP:

    def __init__(self, polygon: Union[Polygon, MultiPolygon], angle_threshold):
        self.polygon = polygon  # The region to cover
        self.angle_threshold = angle_threshold
        self.simple_polygons = PolygonUtils.decompose_polygon(self.polygon, self.angle_threshold)
        self.fov_width = None  # Field of View width of the drone's camera
        self.start_point = 0  # Starting point
        self.end_point = 1  # Ending point

    def _generate_paths(self, generate_path_function, fov_width, angle_degrees=None):
        paths = []
        for i, poly in enumerate(self.simple_polygons):
            path = generate_path_function(poly, fov_width, angle_degrees) if angle_degrees else generate_path_function(
                poly, fov_width)
            if i % 2 != 0:
                path = path[::-1]
            paths.append(path)
        return paths

    def generate_shape_mimicking_spiral(self, fov_width, steps=100):
        shape_mimicking_spirals = [PathPlannerUtils.generate_single_spiral(poly, fov_width, steps) for poly in
                                   self.simple_polygons]

        total_path_length, total_turns, transition_distance, optimized_sequence = calculate_spiral_metrics(
            shape_mimicking_spirals)

        optimized_combined_spirals = [shape_mimicking_spirals[i] for i in optimized_sequence]

        title = (
            f"Shape-Mimicking Spirals\n"
            f"Total Path Length: {total_path_length:.2f} units\n"
            f"Total Turns: {total_turns}\n"
            f"Total Transition Distance: {transition_distance:.2f} units"
        )

        PlottingUtils.plot_combined_lawnmower_path(self.simple_polygons, optimized_combined_spirals, title)

    def compute_rotated_paths_for_polygons(self, fov_width, angle_degrees=0):
        rotated_paths = self._generate_paths(PathPlannerUtils.compute_rotated_path, fov_width, angle_degrees)
        transition_distances = calculate_transition_distances(rotated_paths)
        optimized_sequence = optimize_sequence_nearest_neighbor(rotated_paths, transition_distances)
        total_length, total_turns, total_transition_distance = calculate_metrics(rotated_paths,
                                                                                 optimized_sequence,
                                                                                 transition_distances)

        title = (
            f'Optimized Rotated Path at {angle_degrees} Degrees \n'
            f'Total Path Length: {total_length:.2f} units \n'
            f'Total Number of Turns: {total_turns} \n'
            f'Total Transition Distance: {total_transition_distance:.2f} units'
        )
        optimized_rotated_paths = [rotated_paths[i] for i in optimized_sequence]

        return optimized_rotated_paths, total_length, total_turns, total_transition_distance, title

    def calculate_best_worst_paths(self, paths_rotated, lengths_rotated, total_turns_rotated,
                                   transition_distance):
        min_length_index = np.argmin(lengths_rotated)
        max_length_index = np.argmax(lengths_rotated)

        shortest_details = paths_rotated[min_length_index]
        longest_details = paths_rotated[max_length_index]

        return {
            'shortest': {
                'details': shortest_details,
                'length': lengths_rotated[min_length_index],
                'turns': total_turns_rotated[min_length_index],
                'transition_distance': transition_distance[min_length_index]
            },
            'longest': {
                'details': longest_details,
                'length': lengths_rotated[max_length_index],
                'turns': total_turns_rotated[max_length_index],
                'transition_distance': transition_distance[max_length_index]
            }
        }

    def test_multiple_angles(self, fov_width, step=10):
        paths_rotated = []
        lengths_rotated = []
        total_turns_rotated = []
        transition_distance = []

        angles = np.arange(0, 180, step)

        for angle in angles:
            paths, total_length, total_turns, total_transition_distance, _ = (
                self.compute_rotated_paths_for_polygons(fov_width, angle)
            )
            paths_rotated.append((angle, paths))
            lengths_rotated.append(total_length)
            total_turns_rotated.append(total_turns)
            transition_distance.append(total_transition_distance)

        best_worst_paths = self.calculate_best_worst_paths(paths_rotated, lengths_rotated, total_turns_rotated,
                                                           transition_distance)

        PlottingUtils.visualize_best_worst_paths(self.simple_polygons, best_worst_paths['shortest'],
                                                 best_worst_paths['longest'])

    def test_multiple_angles_for_decomposed_polygons(self, fov_width, angle_step):
        angles = np.arange(0, 180, angle_step)
        best_paths = []
        best_lengths = []

        for polygon in self.simple_polygons:
            best_path, best_length = PathPlannerUtils.find_best_path_for_polygon(polygon, fov_width, angles)
            best_paths.append(best_path)
            best_lengths.append(best_length)

        total_transition_distance, optimized_sequence = calculate_total_transition_distance(best_paths)
        optimized_combined_paths = [best_paths[i] for i in optimized_sequence]
        num_turns = sum(PathPlannerUtils.count_turns_in_path(path) for path in optimized_combined_paths)

        title = (
            f'Optimized Rotated Path with Multiple Angles \n'
            f'Total Path Length: {sum(best_lengths):.2f} units \n'
            f'Total Turns: {num_turns:.2f} units\n'
            f'Total Transition Distance: {total_transition_distance:.2f} units\n'
        )

        PlottingUtils.plot_combined_lawnmower_path(self.simple_polygons, optimized_combined_paths, title=title)
