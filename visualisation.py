import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def plot_irradiance(self, ax, nodes, irradiance_values):
        colors = plt.cm.inferno(np.array(irradiance_values) / max(irradiance_values))
        sc = ax.scatter([node.lon for node in nodes],
                        [node.lat for node in nodes],
                        [node.alt for node in nodes],
                        c=irradiance_values, s=100, cmap='YlOrRd')
        plt.colorbar(sc, ax=ax, label='Irradiance (W/m^2)')

    def plot_wind_vectors(self, ax, coords, wind_directions, wind_speeds):
        quiver_coords = np.array(coords).T
        ax.quiver(quiver_coords[0], quiver_coords[1], quiver_coords[2],
                  wind_directions, wind_speeds, [0] * len(wind_directions),
                  length=0.4, normalize=True, color='blue', label='Wind Vectors')

    def plot_precipitation(self, ax, precipitation_coords, precipitation_sizes):
        if precipitation_coords:
            precipitation_coords = np.array(precipitation_coords).T
            ax.scatter(precipitation_coords[0], precipitation_coords[1], precipitation_coords[2],
                       c='darkgreen', s=precipitation_sizes, marker='v', label='Precipitation')

    def highlight_start_end_nodes(self, ax, nodes):
        ax.scatter(nodes[0].lon, nodes[0].lat, nodes[0].alt, marker='x', color='black', s=100, label='Start Node')
        ax.scatter(nodes[-1].lon, nodes[-1].lat, nodes[-1].alt, marker='x', color='red', s=100, label='End Node')

    def shade_nighttime(self, ax, nighttime_nodes):
        for node in nighttime_nodes:
            ax.plot([node.lon], [node.lat], [node.alt], 'o', markersize=15, color='black', alpha=0)

    def plot_nodes(self, nodes):
        # Apply visual scaling to lat and lon for all nodes
        # scaled_lats = [self.visual_offset(node.lat, node.lon)[0] for node in nodes]
        # scaled_lons = [self.visual_offset(node.lat, node.lon)[1] for node in nodes]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Store data for quiver plot and other visualizations
        wind_directions, wind_speeds, coords, precipitation_coords, precipitation_sizes, irradiance_values = [], [], [], [], [], []

        for node in nodes:
            weather_info = node.weather.asof(node.time)

            # Gather data for various visualizations
            # 1. Color-code based on Total Sun Irradiance
            irradiance_components = ['ghi', 'dhi', 'dni']
            total_irradiance = sum([weather_info[component] for component in irradiance_components])
            irradiance_values.append(total_irradiance)

            # 2. Wind Direction and Strength data for quiver plot
            wind_direction = weather_info['winddirection_100m']
            wind_speed = weather_info['wind_speed']
            wind_dx = wind_speed * np.cos(np.radians(wind_direction))
            wind_dy = wind_speed * np.sin(np.radians(wind_direction))
            wind_directions.append(wind_dx)
            wind_speeds.append(wind_dy)
            coords.append((node.lon, node.lat, node.alt))

            # 3. Precipitation data
            precipitation = weather_info['precipitation']
            if precipitation > 0.1:  # Threshold for showing precipitation
                precipitation_coords.append((node.lon, node.lat, node.alt))
                precipitation_sizes.append(precipitation * 20)  # Adjust multiplier for visualization

        self.plot_irradiance(ax, nodes, irradiance_values)
        self.plot_wind_vectors(ax, coords, wind_directions, wind_speeds)
        self.plot_precipitation(ax, precipitation_coords, precipitation_sizes)
        self.highlight_start_end_nodes(ax, nodes)

        # Nighttime shading
        nighttime_nodes = [node for node in nodes if 21 <= node.time.hour or node.time.hour < 4]
        self.shade_nighttime(ax, nighttime_nodes)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude')
        ax.legend()
        plt.show()

    def plot_graph(self, graph, optimal_path=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract weather data and use helper functions
        wind_directions, wind_speeds, coords, precipitation_coords, precipitation_sizes, irradiance_values = [], [], [], [], [], []

        for node in graph.nodes:
            weather_info = node.weather.asof(node.time)

            # 1. Color-code based on Total Sun Irradiance
            irradiance_components = ['ghi', 'dhi', 'dni']
            total_irradiance = sum([weather_info[component] for component in irradiance_components])
            irradiance_values.append(total_irradiance)

            # 2. Wind Direction and Strength data for quiver plot
            wind_direction = weather_info['winddirection_100m']
            wind_speed = weather_info['wind_speed']
            wind_dx = wind_speed * np.cos(np.radians(wind_direction))
            wind_dy = wind_speed * np.sin(np.radians(wind_direction))
            wind_directions.append(wind_dx)
            wind_speeds.append(wind_dy)
            coords.append((node.lon, node.lat, node.alt))

            # 3. Precipitation data
            precipitation = weather_info['precipitation']
            if precipitation > 0.1:  # Threshold for showing precipitation
                precipitation_coords.append((node.lon, node.lat, node.alt))
                precipitation_sizes.append(precipitation * 20)  # Adjust multiplier for visualization

        self.plot_irradiance(ax, graph.nodes, irradiance_values)
        self.plot_wind_vectors(ax, coords, wind_directions, wind_speeds)
        self.plot_precipitation(ax, precipitation_coords, precipitation_sizes)
        self.highlight_start_end_nodes(ax, [graph.nodes[0], graph.nodes[-1]])

        # If an optimal path is provided, plot it as a line
        if optimal_path is not None:
            xs = [node.lon for node in optimal_path]
            ys = [node.lat for node in optimal_path]
            zs = [node.alt for node in optimal_path]
            ax.plot(xs, ys, zs, color='red', linewidth=2.5)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude')
        ax.legend()

        plt.show()

    def visual_offset(self, lat, lon, scale_factor=1.3):
        """
        Modify lat and lon for visualization purposes.
        """
        lat_offset = lat * scale_factor
        lon_offset = lon * scale_factor
        return lat_offset, lon_offset
