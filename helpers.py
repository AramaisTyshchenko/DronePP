from datetime import timedelta, datetime
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas
import pvlib

from nodes import Node

# Assuming the Earth is a perfect sphere with radius 6371 km
EARTH_RADIUS_KM = 6371


def km_to_degrees(km, latitude):
    """
    Convert distance in km to degrees for a given latitude.
    """
    # Convert latitude from degrees to radians
    latitude_rad = np.radians(latitude)

    # Compute the number of degrees corresponding to the input distance in km
    delta_lat = km / EARTH_RADIUS_KM
    delta_lon = km / (EARTH_RADIUS_KM * np.cos(latitude_rad))

    return delta_lat, delta_lon


def round_time(time_obj: datetime, round_to: int):
    if time_obj.minute >= round_to // 2:
        time_obj = time_obj + timedelta(hours=1)
    time_obj = time_obj.replace(minute=0, second=0)
    return time_obj


def calculate_distance(node1: Node, node2: Node):
    """
    Calculate the straight-line distance between two nodes.

    Parameters:
    - node1 (Node): A Node containing the latitude, longitude, and altitude of the first point.
    - node2 (Node): A Node containing the latitude, longitude, and altitude of the second point.

    Returns:
    - distance (float): The distance between the two nodes, in meters.
    """
    # Radius of the earth in meters
    R = EARTH_RADIUS_KM * 1000

    lat1, lon1, alt1 = radians(node1.lat), radians(node1.lon), node1.alt
    lat2, lon2, alt2 = radians(node2.lat), radians(node2.lon), node2.alt

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    dalt = alt2 - alt1

    # Haversine formula to calculate the distance between the nodes at sea level
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    sea_level_distance = R * c

    # Pythagorean theorem to calculate the straight-line distance in 3D space
    distance = sqrt(sea_level_distance ** 2 + dalt ** 2)
    return distance


def middle_node(node1: Node, node2: Node):
    """
    Function to calculate the midpoint between two nodes on the Earth.

    Parameters:
    - node1 (Node): A Node instance representing the first point.
    - node2 (Node): A Node instance representing the second point.

    Returns:
    - mid_node (Node): A Node instance representing the midpoint between node1 and node2.
    """
    # Converting latitude and longitude from degrees to radians
    lat1, lon1, alt1 = radians(node1.lat), radians(node1.lon), node1.alt
    lat2, lon2, alt2 = radians(node2.lat), radians(node2.lon), node2.alt

    # The longitude difference
    d_lon = lon2 - lon1

    # Calculate the midpoint between the two points on the Earth's surface
    B_x = cos(lat2) * cos(d_lon)
    B_y = cos(lat2) * sin(d_lon)
    lat3 = atan2(sin(lat1) + sin(lat2), sqrt(((cos(lat1) + B_x) ** 2 + B_y ** 2)))
    lon3 = lon1 + atan2(B_y, cos(lat1) + B_x)

    # The altitude difference
    alt3 = (alt1 + alt2) / 2

    # Converting latitude and longitude from radians to degrees
    lat3 = np.degrees(lat3)
    lon3 = np.degrees(lon3)

    mid_node = Node(lat3, lon3, alt3)

    return mid_node


def calculate_tilt_and_azimuth(node1: Node, node2: Node):
    """
    Calculate the tilt and azimuth from node1 to node2.

    Parameters:
    - node1 (Node): A Node instance representing the first point.
    - node2 (Node): A Node instance representing the second point.

    Returns:
    - tilt (float): The tilt from node1 to node2, in degrees.
    - azimuth (float): The azimuth from node1 to node2, in degrees.
    """
    # Converting latitude and longitude from degrees to radians
    lat1, lon1 = radians(node1.lat), radians(node1.lon)
    lat2, lon2 = radians(node2.lat), radians(node2.lon)

    # The differences
    dlon = lon2 - lon1
    dalt = node2.alt - node1.alt

    # Calculate azimuth
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    azimuth = atan2(y, x)

    # Calculate distance
    distance = calculate_distance(node1, node2)

    # Calculate tilt
    tilt = atan2(dalt, distance)

    # Convert from radians to degrees
    tilt = np.degrees(tilt)
    azimuth = (np.degrees(azimuth) + 360) % 360  # Normalize to 0-360

    return tilt, azimuth


def calculate_optimal_tilt_and_azimuth(lat, lon, time: pandas.DatetimeIndex):
    """
    Calculate the optimal tilt and azimuth to align with the sun.

    Parameters:
    - node (Node): A Node instance representing the location.
    - time (datetime): A datetime instance representing the time.

    Returns:
    - tilt (float): The optimal tilt, in degrees.
    - azimuth (float): The optimal azimuth, in degrees.
    """
    # Calculate the solar position
    solar_position = pvlib.solarposition.get_solarposition(time, lat, lon)

    # Extract the solar altitude and azimuth
    solar_altitude = solar_position['apparent_elevation'].values[0]
    solar_azimuth = solar_position['azimuth'].values[0]

    # Calculate the optimal tilt and azimuth
    tilt = 90 - solar_altitude
    azimuth = solar_azimuth

    return tilt, azimuth
