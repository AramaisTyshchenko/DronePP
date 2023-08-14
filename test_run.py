import asyncio
from datetime import datetime

from mavsdk import System
from mavsdk.mission import MissionPlan, MissionItem

from costs import Cost
from drone_telemetry import DroneState
from nodes import Node
from search import a_star, greedy_best_first_search


async def run():
    start_node = Node(18.5567, -89.074, 100, time=datetime(2020, 7, 1, 10, 0, 0), name='start_node')
    end_node = Node(25.7567, -95.894, 100, name='end_node')
    flight_cost = Cost(start_node, end_node, step_sizes=(1, 1, 50))
    # flight_cost = Cost(start_node, end_node, step_sizes=(0.5, 0.5, 50))

    graph = flight_cost.generate_graph()

    dijkstra_path = flight_cost.calculate_dijkstra_path(graph)
    a_star_path = a_star(flight_cost.start_node, flight_cost.end_node, graph)
    greedy_path = greedy_best_first_search(flight_cost.start_node, flight_cost.end_node, graph)

    flight_cost.visualizer.plot_graph(graph, dijkstra_path)
    flight_cost.visualizer.plot_graph(graph, a_star_path)

    # # Run the Genetic Algorithm
    # ga = GeneticAlgorithm(graph, flight_cost.start_node, flight_cost.end_node)
    # ga_path = ga.run()



    drone = System(mavsdk_server_address='localhost', port=50051)
    await drone.connect(system_address="udp://:14540")

    # Define mission items
    mission_items = [
        MissionItem(20.0077, -90.0743, 15, 10, True, 0, 0, MissionItem.CameraAction.NONE, 0, 0, 1, 0, 0),
        MissionItem(20.0030, -90.0843, 20, 10, True, 0, 0, MissionItem.CameraAction.NONE, 0, 0, 1, 0, 0),
        MissionItem(20.0000, -90.074, 15, 10, True, 0, 0, MissionItem.CameraAction.NONE, 0, 0, 1, 0, 0),
    ]

    mission_plan = MissionPlan(mission_items)
    await drone.mission.upload_mission(mission_plan)

    print("Starting mission...")
    await drone.action.arm()
    await drone.mission.start_mission()

    # later, in your run function...
    state = DroneState()
    update_task = asyncio.create_task(state.update_position(drone))
    update_task2 = asyncio.create_task(state.update_battery(drone))

    # ...
    # when you need to access the position:
    while True:
        await asyncio.sleep(1)

        print(state.position)
        print(state.update_euler_angle)
        print(state.battery)

    # Remember to cancel the task when you're done with it
    update_task.cancel()
    update_task2.cancel()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
