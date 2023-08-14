from datetime import timedelta

import networkx as nx
import numpy as np

from helpers import calculate_distance
from nodes import Node
from pvsystem import PVSystemSimulator
from visualisation import Visualizer
from weather import WeatherData


class Cost:

    def __init__(self, start_node: Node, end_node: Node, step_sizes: tuple = (0.3, 0.3, 100)):
        self.distance_coeff = 0
        self.pvsystem_coeff = 1
        self.grid_extension = 0
        self.drone = None
        self.start_node = start_node
        self.end_node = end_node
        # Define step sizes for lat, lon, and height
        self.step_sizes = np.array(step_sizes)
        self.speed = 12
        self.weather_data = WeatherData()
        self.pv_system = PVSystemSimulator()
        self.visualizer = Visualizer()

    @staticmethod
    def calculate_computational_load(nodes):
        num_nodes = len(nodes)
        # Calculate expected number of edges
        expected_edges = num_nodes * (num_nodes - 1) // 2
        distance = calculate_distance(nodes[0], nodes[1])
        print(f"The graph will have approximately {expected_edges} edges. Nodes length = {len(nodes)},"
              f"flight distance = {distance},"
              f"start node = {nodes[0].info()},"
              f"end node = {nodes[1].info()}")

    def sun_power_output(self, node1: Node, node2: Node):
        weather = self.weather_data.get_weather_for_edge(node1, node2)
        power_output = self.pv_system.power_output(weather, node1, node2)
        return power_output

    def cost(self, node1: Node, node2: Node):
        """

        :param node1:
        :param node2:
        :return: cost
        """
        # # Add flight constraints to the cost
        # if node2.speed > self.MAX_SPEED:
        #     return np.inf  # Infinite cost if the speed constraint is violated

        distance = calculate_distance(node1, node2)
        power_output_cost = abs(self.sun_power_output(node1, node2))
        distance_cost = abs(distance)
        return power_output_cost * self.pvsystem_coeff + distance_cost * self.distance_coeff

    def generate_grid_of_nodes(self):
        # Define start and end nodes with extension
        start_point = np.array(
            [self.start_node.lat, self.start_node.lon, self.start_node.alt]) - self.step_sizes * self.grid_extension
        end_point = np.array(
            [self.end_node.lat, self.end_node.lon, self.end_node.alt]) + self.step_sizes * self.grid_extension

        # Compute number of steps needed for each dimension
        n_steps = np.ceil(np.abs((end_point - start_point) / self.step_sizes)).astype(int) + 1
        # Generate grid points for each dimension
        grid_points = [np.linspace(start, end, steps) for start, end, steps in
                       zip(start_point, end_point, n_steps)]
        # Generate grid
        grid = np.stack(np.meshgrid(*grid_points), -1).reshape(-1, 3)
        # Adjust the end node's time
        self.end_node.time = self.start_node.time + timedelta(seconds=
                                                              calculate_distance(self.start_node, self.end_node)
                                                              / self.speed)
        self.start_node.weather = self.weather_data.get_weather('open_meteo', self.start_node)
        self.end_node.weather = self.weather_data.get_weather('open_meteo', self.end_node)

        node_list = [self.start_node, self.end_node]

        shortest_distance = np.inf
        for point in grid:
            lat, lon, alt = point
            # Create a Node for the current point
            node = Node(lat=lat, lon=lon, alt=alt)
            # Calculate the distance from the start node to the current node
            distance = calculate_distance(self.start_node, node)
            if distance and distance < shortest_distance:
                shortest_distance = distance
            # Calculate the travel time from the start node to the current node
            travel_time = distance / self.speed
            # Adjust the time of the current node
            node.time = self.start_node.time + timedelta(seconds=travel_time)
            # Fetch weather data for the current node
            # Assign the weather data to the node attributes
            node.weather = self.weather_data.get_weather('open_meteo', node)
            # Append the node to the list
            node_list.append(node)

        self.start_node.shortest_distance = shortest_distance
        # # Filter nodes based on flight constraints
        # node_list = [node for node in node_list if self.MIN_ALTITUDE <= node.alt <= self.MAX_ALTITUDE]
        # Calculate computational load
        self.calculate_computational_load(node_list)
        return node_list

    def generate_graph(self):
        # Generate the grid of nodes
        nodes = self.generate_grid_of_nodes()
        self.visualizer.plot_nodes(nodes)
        # Initialize an empty graph
        graph = nx.Graph()

        # Maximum allowable distance for creating an edge
        max_distance = nodes[0].shortest_distance * 2.5  # example: half an hour flight distance at max speed

        # Add the nodes to the graph
        for node in nodes:
            graph.add_node(node)

        # Add the edges to the graph with constraints
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    distance = calculate_distance(node1, node2)
                    if distance <= max_distance:
                        weight = self.cost(node1, node2)
                        graph.add_edge(node1, node2, weight=weight)

        return graph

    def calculate_dijkstra_path(self, graph):
        # Find the shortest path in the graph from start_node to end_node
        optimal_path = nx.dijkstra_path(graph, self.start_node, self.end_node, weight='weight')
        return optimal_path

    def calculate_speed_cost(self, speed):
        optimum_speed = self.speed
        speed_weight = 1.0
        return speed_weight * (speed - optimum_speed) ** 2

    def calculate_altitude_cost(self, altitude):
        altitude_weight = 0.1
        return altitude_weight * altitude

    def calculate_tilt_cost(self, tilt):
        tilt_weight = 1.0
        return tilt_weight * tilt

    def calculate_soc_penalty(self, battery_soc):
        soc_penalty = 0
        if battery_soc < 0.25 or battery_soc > 0.9:
            soc_penalty = 1000  # Placeholder
        return soc_penalty

    def calculate_weather_factor(self, weather_conditions):
        weather_factor = 1
        if weather_conditions == 'windy':
            weather_factor = 1.2
        elif weather_conditions == 'rainy':
            weather_factor = 1.3
        return weather_factor

    def calculate_maneuver_factor(self, maneuver):
        maneuver_factor = 1
        if maneuver == 'ascent':
            maneuver_factor = 1.2
        elif maneuver == 'descent':
            maneuver_factor = 1.1
        elif maneuver == 'acceleration':
            maneuver_factor = 1.3
        return maneuver_factor

    def calculate_photo_cost(self, photo_conditions):
        photo_weight = 1.0
        photo_cost = 0
        if photo_conditions['altitude'] > 200:  # Altitude is above 200 meters
            photo_cost += photo_weight
        if photo_conditions['weather'] != 'clear':
            photo_cost += photo_weight
        return photo_cost

    def calculate_wind_cost(self, wind_conditions):
        wind_weight = 1.0
        return wind_weight * (wind_conditions['speed'] + wind_conditions['turbulence'])

    def cost_other(self, battery_soc, speed, altitude, tilt, weather_conditions, maneuver, photo_conditions,
                   wind_conditions):
        speed_cost = self.calculate_speed_cost(speed)
        altitude_cost = self.calculate_altitude_cost(altitude)
        tilt_cost = self.calculate_tilt_cost(tilt)
        soc_penalty = self.calculate_soc_penalty(battery_soc)
        weather_factor = self.calculate_weather_factor(weather_conditions)
        maneuver_factor = self.calculate_maneuver_factor(maneuver)
        photo_cost = self.calculate_photo_cost(photo_conditions)
        wind_cost = self.calculate_wind_cost(wind_conditions)

        cost = ((speed_cost + altitude_cost + tilt_cost) * weather_factor * maneuver_factor +
                soc_penalty + photo_cost + wind_cost)
        return cost

    # def update_graph(self, graph):
    #     # Update the node attributes and edge weights in the graph
    #     for node1, node2, data in graph.edges(data=True):
    #         # Update node attributes (like weather conditions) based on new data
    #         # node1.weather = get_new_weather(node1)
    #         # node2.weather = get_new_weather(node2)
    #         # Recalculate the edge weight
    #         data['weight'] = self.cost(node1, node2)
    #
    # def execute_path(self, graph, start_node, end_node):
    #     # Calculate the initial optimal path
    #     optimal_path = self.calculate_optimal_path(graph, start_node, end_node)
    #
    #     # Start executing the path with the drone
    #     for i in range(len(optimal_path) - 1):
    #         current_node = optimal_path[i]
    #         next_node = optimal_path[i + 1]
    #
    #         # Move the drone from current_node to next_node
    #         # self.drone.move_to(next_node)
    #
    #         # Periodically update the graph and recalculate the optimal path
    #         if i % 10 == 0:  # For example, every 10 steps
    #             self.update_graph(graph)
    #             optimal_path = self.calculate_optimal_path(graph, current_node, end_node)


def calculate_energy_consumption(self, speed, weather_conditions, maneuver):
    # Constants
    g = 9.81  # gravitational constant in m/s^2
    rho = 1.225  # air density at sea level in kg/m^3
    Cd = 0.1  # drag coefficient, this is a rough estimate and the actual value can vary

    # Lift energy
    lift_energy = 0.5 * self.weight * g * speed / self.propeller_efficiency

    # Drag energy
    drag_energy = 0.5 * Cd * rho * self.surface_area * speed ** 3 / self.motor_efficiency

    # Onboard systems energy
    # This would depend on the specific systems on your drone
    # For now, let's assume it's a constant value
    systems_energy = 10  # in watts

    # Weather conditions can affect both lift and drag energy
    # For simplicity, let's assume a linear relationship
    weather_factor = 1
    if weather_conditions == 'windy':
        weather_factor = 1.2
    elif weather_conditions == 'rainy':
        weather_factor = 1.3

    # Flight maneuvers can increase energy consumption
    maneuver_factor = 1
    if maneuver == 'ascent':
        maneuver_factor = 1.2
    elif maneuver == 'descent':
        maneuver_factor = 1.1
    elif maneuver == 'acceleration':
        maneuver_factor = 1.3

    total_energy = (lift_energy + drag_energy + systems_energy) * weather_factor * maneuver_factor

    return total_energy

# Clear sky
# dni, ghi, dhi = 800, 800, 800
# airmass_relative = 1
# absolute_airmass = atmosphere.get_absolute_airmass(airmass_relative)
# Make a run for the next python code with the  start_node with (lat=20.0567, lon=-90.074, alt=100)
# and end_node with (lat=20.7567, lon=-90.294, alt=700)
