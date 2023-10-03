class DroneState:
    def __init__(self):
        self.euler_angle = None
        self.position = None
        self.battery = None

    async def update_position(self, drone):
        async for position in drone.telemetry.position():
            self.position = position

    async def update_euler_angle(self, drone):
        async for euler_angle in drone.telemetry.eulerAngle():
            self.euler_angle = euler_angle

    async def update_battery(self, drone):
        async for battery in drone.telemetry.battery():
            self.battery = battery

    async def update_mission_progress(self, drone):
        async for mission_progress in drone.mission.mission_progress():
            self.current_mission_item = mission_progress.current
            self.total_mission_items = mission_progress.total
