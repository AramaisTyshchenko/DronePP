from datetime import datetime


class Node:
    def __init__(self, lat, lon, alt, time=datetime(1991, 6, 1, 0, 0, 0), name=None):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.time: datetime = time
        # self.tilt: float = 0
        # self.azimuth: float = 0
        self.speed: float = 12
        self.battery: float = 100
        self.cloudiness: float = 0
        self.wind: float = 0
        self.danger: float = 0
        self.loiter_time: float = 0
        self.weather = 0
        self.name: str = name

    # def __eq__(self, other):
    #     if not isinstance(other, Node):
    #         # don't attempt to compare against unrelated types
    #         return NotImplemented
    #
    #     return self.lat == other.lat and self.lon == other.lon and self.alt == other.alt and self.time == other.time
    #
    # def __hash__(self):
    #     return hash((self.lat, self.lon, self.alt, self.time))

    def info(self):
        print(
            f"Node(\n"
            f"\tLatitude: {self.lat}\n"
            f"\tLongitude: {self.lon}\n"
            f"\tAltitude: {self.alt}\n"
            # f"\tTilt: {self.tilt}\n"
            # f"\tAzimuth: {self.azimuth}\n"
            f"\tSpeed: {self.speed}\n"
            f"\tBattery: {self.battery}\n"
            f"\tCloudiness: {self.cloudiness}\n"
            f"\tWind: {self.wind}\n"
            f"\tDanger: {self.danger}\n"
            f"\tTime: {self.time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"\tName: {self.name}\n"
            f")"
        )
