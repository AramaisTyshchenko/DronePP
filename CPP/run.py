import numpy as np

from CPP.cpp import CPP
from CPP.cpp_utils import PolygonUtils


def runner():
    # # Generate random latitude and longitude points
    # num_points = 7
    # lat_points, lon_points = generate_random_lat_lon_points(num_points=num_points)

    # Generate random points
    # num_points = 100
    # points = np.random.rand(num_points, 2) * 10
    points = np.array([[5.61829623, 9.21979401],
                       [5.42486876, 7.81504775],
                       [6.29762845, 2.3618226],
                       [7.78963672, 0.84733478],
                       [9.7166186, 4.64988229],
                       [7.39075966, 1.62916138],
                       [9.02337881, 9.57842792],
                       [8.96202164, 7.37757383],
                       [8.68207414, 3.5113755],
                       [5.85102206, 7.12886857],
                       [7.65261707, 7.28061118],
                       [2.74252511, 0.92671862],
                       [5.53876342, 6.15245035],
                       [5.72052669, 5.90022148],
                       [3.65373689, 2.92831709],
                       [1.99152343, 1.03387227],
                       [3.83363723, 0.99600903],
                       [1.9425715, 2.35179429],
                       [0.12590851, 6.84375743],
                       [7.57463905, 1.82488995],
                       [1.8054016, 2.46825497],
                       [3.05042975, 9.09325728],
                       [7.64080185, 5.23394346],
                       [4.28217563, 2.82933748],
                       [0.03274719, 0.63312357],
                       [0.20325908, 1.28114276],
                       [3.80967673, 0.35805932],
                       [4.87035637, 9.804241],
                       [1.17684493, 7.12837824],
                       [7.95542535, 0.543549],
                       [7.99161969, 0.07527494],
                       [5.46585412, 2.03359456],
                       [1.04500129, 7.7825045],
                       [1.85285529, 2.27724185],
                       [2.51758559, 3.15619993],
                       [0.52411507, 8.92579154],
                       [4.38851882, 2.75879062],
                       [4.43244082, 8.68990611],
                       [6.22129977, 1.25325567],
                       [4.6356246, 4.50613817],
                       [1.88412423, 5.86568526],
                       [6.18871956, 9.55007612],
                       [3.50853075, 7.85941543],
                       [6.74544708, 3.91021035],
                       [7.57610212, 3.23501669],
                       [6.7522467, 7.87928299],
                       [5.16379542, 6.56474371],
                       [2.71831095, 1.59442228],
                       [7.64073404, 3.32912156],
                       [4.70831398, 8.6993607],
                       [3.18690654, 6.67081064],
                       [3.07915569, 4.42248622],
                       [0.62886563, 0.19708849],
                       [2.35810102, 0.50089796],
                       [5.01552841, 8.51683392],
                       [0.39779613, 6.68464452],
                       [5.69630545, 2.2917004],
                       [0.79790143, 1.18124512],
                       [0.55439337, 4.50468029],
                       [2.09022333, 5.33600879],
                       [1.93133718, 9.33591055],
                       [1.55632285, 6.77937225],
                       [7.35783372, 6.12280332],
                       [5.89099915, 8.99836244],
                       [9.36603312, 6.30796878],
                       [9.24276119, 1.2738271],
                       [3.76694311, 5.26317891],
                       [4.12084712, 5.87488075],
                       [5.58292701, 0.79715812],
                       [7.56927439, 2.72981215],
                       [0.58364034, 6.05304254],
                       [6.56342142, 2.51837725],
                       [9.68026149, 5.69119922],
                       [9.45584196, 1.7510664],
                       [6.45685785, 0.18595681],
                       [2.70406245, 3.8095074],
                       [4.49139964, 3.30628412],
                       [6.11129416, 7.76497594],
                       [0.88177514, 5.34514844],
                       [1.12695483, 0.67257368],
                       [2.17128272, 8.23802004],
                       [3.89119665, 1.37708708],
                       [9.24016775, 9.3791441],
                       [1.08496599, 9.44314103],
                       [3.78047387, 4.21206002],
                       [7.33832049, 0.07411702],
                       [5.47499633, 1.01903342],
                       [9.12000982, 7.72402894],
                       [7.4864434, 5.59440619],
                       [5.8868934, 1.90648156],
                       [6.29272491, 8.57680904],
                       [0.62883242, 4.19791848],
                       [5.03944896, 9.49746577],
                       [9.365397, 8.20544877],
                       [6.57858328, 2.09598626],
                       [3.54912201, 9.32444167],
                       [5.84347767, 3.6405892],
                       [9.58200817, 5.87864065],
                       [5.5950429, 1.41511635],
                       [7.28396864, 9.11053868]])
    # Compute the alpha shape
    alpha = 0.7
    fov_width = 0.00003
    fov_width = 0.1
    angle_threshold = 180

    concave_polygon = PolygonUtils.alpha_shape(points, alpha)
    cpp = CPP(concave_polygon, angle_threshold)
    # Decompose the polygon
    simple_polygons = PolygonUtils.decompose_polygon(concave_polygon, angle_threshold=360)
    # Plot decomposed polygons
    # PlottingUtils.plot_decomposed_polygons(simple_polygons)

    ############################################################
    ############################################################
    concave_polygon = simple_polygons[4]
    cpp = CPP(concave_polygon, angle_threshold)
    # Decompose the polygon
    simple_polygons = cpp.simple_polygons
    # # Plot decomposed polygons
    # PlottingUtils.plot_decomposed_polygons(simple_polygons)

    simple_polygons1 = PolygonUtils.decompose_polygon(concave_polygon, angle_threshold=360)
    # simple_polygons2 = PolygonUtils.extract_largest_convex_polygon(concave_polygon, min_angle=60)
    # PlottingUtils.plot_decomposed_polygons(simple_polygons1)
    # PlottingUtils.plot_decomposed_polygons([simple_polygons2])

    ############################################################
    # Compute the lawnmower paths for the decomposed polygons

    # Generate shape-mimicking spirals for each decomposed polygon
    # cpp.generate_shape_mimicking_spiral(fov_width=fov_width, angle_threshold=360, steps=100)

    # optimized_rotated_paths, _, _, _, title = (
    #     cpp.compute_rotated_paths_for_polygons(fov_width, 30))
    # PlottingUtils.plot_combined_lawnmower_path(simple_polygons, optimized_rotated_paths, title)
    #
    # # Test for multiple angles and find the shortest path using the rotation approach
    # cpp.test_multiple_angles(fov_width=fov_width, step=10)
    cpp.test_multiple_angles_for_decomposed_polygons_enhanced(fov_width=fov_width, angle_step=1)


# Run the functions
runner()
