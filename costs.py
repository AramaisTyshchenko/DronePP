from datetime import timedelta

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from helpers import calculate_distance
from nodes import Node
from pvsystem import PVSystemSimulator
from weather import WeatherData


class Cost:
    MAX_ALTITUDE = 1000  # Maximum altitude in meters
    MIN_ALTITUDE = 50  # Minimum altitude in meters
    MAX_SPEED = 20  # Maximum speed in m/s
    MAX_TILT = 30  # Maximum tilt in degrees

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

    @staticmethod
    def calculate_computational_load(nodes):
        num_nodes = len(nodes)
        # Calculate expected number of edges
        expected_edges = num_nodes * (num_nodes - 1) // 2
        print(f"The graph will have approximately {expected_edges} edges. Nodes length = {len(nodes)}")

    def sun_power_output(self, node1: Node, node2: Node):
        weather = self.weather_data.get_weather_for_edge('open_meteo', node1, node2)
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

        # tilt, _ = calculate_tilt_and_azimuth(node1, node2)
        # if tilt > self.MAX_TILT:
        #     return np.inf  # Infinite cost if the tilt constraint is violated

        # # Calculate the travel time from node1 to node2
        # travel_time = distance / self.speed
        # # Adjust the time of node2
        # node2.time = node1.time + timedelta(seconds=travel_time)

        distance = calculate_distance(node1, node2)
        power_output_cost = 1/(0.1 + abs(self.sun_power_output(node1, node2)))
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
        # Create Node objects

        self.end_node.time = self.start_node.time + timedelta(seconds=
                                                              calculate_distance(self.start_node, self.end_node)
                                                              / self.speed)

        node_list = [self.start_node, self.end_node]
        for point in grid:
            lat, lon, alt = point
            # Create a Node for the current point
            node = Node(lat=lat, lon=lon, alt=alt)
            # Calculate the distance from the start node to the current node
            distance = calculate_distance(self.start_node, node)
            # Calculate the travel time from the start node to the current node
            travel_time = distance / self.speed
            # Adjust the time of the current node
            node.time = self.start_node.time + timedelta(seconds=travel_time)
            # Append the node to the list
            node_list.append(node)

        # # Filter nodes based on flight constraints
        # node_list = [node for node in node_list if self.MIN_ALTITUDE <= node.alt <= self.MAX_ALTITUDE]
        self.calculate_computational_load(node_list)
        return node_list

    def plot_nodes(self, nodes):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = [node.lon for node in nodes]
        ys = [node.lat for node in nodes]
        zs = [node.alt for node in nodes]

        ax.scatter(xs, ys, zs)

        # Highlight start and end nodes
        ax.scatter(self.start_node.lon, self.start_node.lat, self.start_node.alt, color='r')
        ax.scatter(self.end_node.lon, self.end_node.lat, self.end_node.alt, color='g')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude')

        plt.show()

    def plot_graph(self, graph, optimal_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot all nodes in the graph
        for node in graph.nodes:
            ax.scatter(node.lon, node.lat, node.alt, color='blue')

        # If an optimal path is provided, plot it with a different color
        if optimal_path is not None:
            for node in optimal_path:
                ax.scatter(node.lon, node.lat, node.alt, color='red')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude')

        plt.show()

    def plot_graph2(self, graph, optimal_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot all nodes in the graph
        for node in graph.nodes:
            ax.scatter(node.lon, node.lat, node.alt, color='blue')

        # If an optimal path is provided, plot it as a line
        if optimal_path is not None:
            xs = [node.lon for node in optimal_path]
            ys = [node.lat for node in optimal_path]
            zs = [node.alt for node in optimal_path]
            ax.plot(xs, ys, zs, color='red')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude')

        plt.show()

    def generate_graph(self):
        # Generate the grid of nodes
        nodes = self.generate_grid_of_nodes()
        self.plot_nodes(nodes)
        # Initialize an empty graph
        graph = nx.Graph()

        # Add the nodes to the graph
        for node in nodes:
            graph.add_node(node)

        # Add the edges to the graph
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    # The cost function is used to calculate the weight of the edge
                    weight = self.cost(node1, node2)
                    graph.add_edge(node1, node2, weight=weight)

        return graph

    def calculate_optimal_path(self, graph):
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
