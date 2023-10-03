import asyncio
import time
from datetime import datetime

from geopy.distance import geodesic
from mavsdk import System

from CPP.cpp import CPP
from CPP.cpp_utils import PolygonUtils, PlottingUtils, PolygonGenerator
from CPP.helpers import create_mission_from_path, compute_utm_zone_number
from CPP.test_data import LAT_LON_POINTS, TEST_SIMPLE_POINTS, LAT_LON_POINTS2
from drone_telemetry import DroneState


# optimized_rotated_paths, _, _, _, title = (
#     cpp.compute_rotated_paths_for_polygons(fov_width, 30))
# PlottingUtils.plot_combined_lawnmower_path(simple_polygons, optimized_rotated_paths, title)
# # Test for multiple angles and find the shortest path using the rotation approach
# cpp.test_multiple_angles(fov_width=fov_width, step=10)

############################################################
# simple_polygons1 = PolygonUtils.decompose_polygon(concave_polygon, angle_threshold=360)
# simple_polygons2 = PolygonUtils.extract_largest_convex_polygon(concave_polygon, min_angle=60)
# PlottingUtils.plot_decomposed_polygons(simple_polygons1)
# PlottingUtils.plot_decomposed_polygons([simple_polygons2])

async def runner():
    # Generate random points
    # points = np.random.rand(100, 2) * 10

    # Compute the alpha shape
    alpha = 0.7
    fov_width = 0.1
    points = TEST_SIMPLE_POINTS

    ############################################################
    # Generate random latitude and longitude points
    # points = PolygonGenerator.generate_random_lat_lon_points(num_points=30)
    # Compute the alpha shape for lat and lon
    # alpha = 0.0000001
    # fov_width = 0.00007
    # points = LAT_LON_POINTS2

    angle_threshold = 240

    concave_polygon = PolygonUtils.alpha_shape(points, alpha)
    cpp = CPP(concave_polygon, angle_threshold)
    # Decompose the polygon
    simple_polygons = PolygonUtils.decompose_polygon(concave_polygon, angle_threshold=360)
    # Plot decomposed polygons
    # PlottingUtils.plot_decomposed_polygons(simple_polygons)

    ############################################################
    concave_polygon = simple_polygons[1]
    cpp = CPP(concave_polygon, angle_threshold)
    # Decompose the polygon
    simple_polygons = cpp.simple_polygons
    # # Plot decomposed polygons
    PlottingUtils.plot_decomposed_polygons(simple_polygons)

    ############################################################
    ############################################################

    # Compute the lawnmower paths for each decomposed polygon
    optimized_paths = cpp.test_multiple_angles_for_decomposed_polygons_enhanced(fov_width=fov_width, angle_step=1)
    # Generate shape-mimicking spirals for each decomposed polygon
    combined_spirals = cpp.generate_shape_mimicking_spiral(fov_width=fov_width, angle_threshold=190, steps=100)

    path = combined_spirals  # This is the path generated from CPP algorithm

    ############################################################
    ############################################################
    ############################################################
    # zone_number = compute_utm_zone_number(points[0][1])
    # # Define mission items
    # # Convert path to mission and upload to the drone
    # mission_plan, mission_item_positions = create_mission_from_path(path)
    # drone = System(mavsdk_server_address='localhost', port=50051)
    # await drone.connect(system_address="udp://:14540")
    # # Clear previous mission
    # # await drone.mission.clear_mission()
    # # time.sleep(3)
    # await drone.mission.upload_mission(mission_plan)
    #
    # print("Starting mission...")
    # await drone.action.arm()
    # await drone.mission.start_mission()
    # # Record mission start time
    # mission_start_time = datetime.now()
    #
    # state = DroneState()
    # update_task = asyncio.create_task(state.update_position(drone))
    # update_task2 = asyncio.create_task(state.update_battery(drone))
    # update_task3 = asyncio.create_task(state.update_mission_progress(drone))
    #
    # # Now you can access the latitude and longitude of the last mission item like this:
    # # last_mission_item_lat, last_mission_item_lon =
    # # last_mission_item_position = (
    # #     last_mission_item_lat,
    # #     last_mission_item_lon
    # # )
    #
    # # when you need to access the position:
    # print(points)
    # while True:
    #     await asyncio.sleep(1)
    #
    #     print(state.position)
    #     print(state.update_euler_angle)
    #     print(state.battery)
    #
    #     current_position = (state.position.latitude_deg, state.position.longitude_deg)
    #     distance_to_last_mission_item = geodesic(mission_item_positions[-1], current_position).meters
    #
    #     # Check if the drone is near the last mission item position and has completed all mission items
    #     if (distance_to_last_mission_item < 10  # Assume 5 meters as a close enough distance
    #             and state.current_mission_item == state.total_mission_items - 1):
    #         break
    #
    #     await asyncio.sleep(1)
    #
    # # Record mission end time
    # mission_end_time = datetime.now()
    # # Calculate and print actual mission time
    # actual_mission_time = mission_end_time - mission_start_time
    # print(f"Actual mission time: {actual_mission_time}")
    #
    # # Remember to cancel the task when you're done with it
    # # Cancel update tasks
    # update_task.cancel()
    # update_task2.cancel()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(runner())
