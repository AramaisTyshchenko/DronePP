import asyncio

from mavsdk import System
from mavsdk.mission import MissionPlan


async def clean_up():
    drone = System(mavsdk_server_address='localhost', port=50051)
    await drone.connect(system_address="udp://:14540")
    # Clear previous mission
    # await drone.mission.clear_mission()
    empty_mission = MissionPlan([])
    await drone.mission.upload_mission(empty_mission)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(clean_up())
