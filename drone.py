class Drone:
    def __init__(self, lat, lon, alt, tilt, azimuth, roll, pitch, yaw, speed, efficiency=1,
                 battery_capacity=1, motor_efficiency=1, propeller_efficiency=1):
        self.max_height = 1000
        self.min_height = 10
        self.opt_speed = 12
        self.weight = 40
        self.surface_area = 0.7

        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.tilt = tilt
        self.azimuth = azimuth
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.speed = speed

        self.efficiency = efficiency
        self.battery_capacity = battery_capacity
        self.battery_level = self.battery_capacity
        self.motor_efficiency = motor_efficiency
        self.propeller_efficiency = propeller_efficiency

    def cloudness(self, weather_data, irradiance):
        # Get weather data from API
        # Adjust irradiance for cloud cover
        cloud_cover = weather_data['cloud_cover']  # Fraction of sky covered by clouds
        irradiance *= (1 - cloud_cover)
