import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString

from lawnmower_path import generate_random_hexagon


# Generate a lawnmower path for coverage
def lawnmower_path(x_hex, y_hex, fov_width):
    polygon = Polygon(zip(x_hex, y_hex))

    min_x, min_y, max_x, max_y = polygon.bounds
    lines = [LineString([(min_x, y), (max_x, y)]) for y in np.arange(min_y, max_y, fov_width)]

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



# Generate the random hexagon
x_hex, y_hex = generate_random_hexagon()

# Define the FOV width and generate the coverage path
fov_width = 0.2
path = lawnmower_path(x_hex, y_hex, fov_width)

# Plot the hexagon and the path
x_path, y_path = zip(*path)