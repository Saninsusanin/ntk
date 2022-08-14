import os
import json
import logging
import numpy as np
import pandas as pd
import logging.config
import networkx as nx
import scipy.stats as st
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm
from math import ceil
from functools import partial
from typing import Union, List
from scipy.spatial import Delaunay
from sklearn.mixture import GaussianMixture
from itertools import product, combinations, chain
from scipy.optimize import minimize as func_minimize
from collections import namedtuple, Counter, defaultdict
from mip import Model, BINARY, xsum, minimize, OptimizationStatus
from networkx.algorithms.community import louvain_communities as louvain

# logging
logging.basicConfig(filename='log.log', level=logging.DEBUG)

# constants
FORMAT = 'svg'
RADIUS = 2000.
PATH = r'/Users/aleksandr_viatkin/PycharmProjects/ntc/Скважины-кусты_Спорышевское мр..xlsx'
BUSH_COST = 100  # mln rubles
BASE_DRILL_COST = 0  # mln rubles
DRILL_METER_COST = 1  # mln rubles
MAX_NUM_OF_WELLS_ON_BUSH = 8
MAX_TIME = 7200
DENOMINATOR = 10
NUMBER_OF_SAMPLES = 100
PROPORTION = 1.8
MAX_POINT_SIZE = 6
PATH_TO_DATA_FOLDER = 'data'
PATH_TO_PLOTS_FOLDER = 'plots'
ITERATIONS_NUM = 1

max_clusters = 1

mpl.rcParams['savefig.format'] = FORMAT


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MyMinMaxScaler:
    __slots__ = ['min', 'max']

    def __init__(self):
        self.max = np.inf
        self.min = np.array([-np.inf])

    def fit(self, points):
        self.min = np.min(points, axis=0)
        self.max = max(np.max(points, axis=0) - self.min)

    def transform(self, points):
        return np.apply_along_axis(lambda point: (point - self.min) / self.max, 1, points)

    def fit_transform(self, points):
        self.fit(points)

        return self.transform(points)

    def reset_radius(self, radius):
        return radius / self.max


def euclidean_distance(first_point: np.ndarray, second_point: np.ndarray, power):
    return (np.sum(np.fabs(first_point - second_point)**power))**(1 / power)


def get_edge(first_id, first_point, second_id, second_point, radius, power):
    if euclidean_distance(first_point, second_point, power) <= radius:
        return first_id, second_id


def build_graph(ids, points, radius, power):
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    edges_generator = filter(lambda x: x is not None,
                             (get_edge(*first_info, *second_info, radius, power)
                              for first_info, second_info in combinations(zip(ids, points), 2)))
    graph.add_edges_from(list(edges_generator))

    return graph


def get_cluster_ids(ids, communities):
    tmp_color_dict = {}

    for color_id, community in enumerate(communities):
        tmp_color_dict.update(dict(zip(community, [color_id] * len(community))))

    return [tmp_color_dict[_id] for _id in ids]


dist = partial(euclidean_distance, power=2)


def get_all_distances(wells):
    return [dist(first_point, second_point) for first_point, second_point in combinations(wells, 2)]


def get_grid_step(wells):
    distances = get_all_distances(wells)
    global DENOMINATOR

    return np.median(distances) / DENOMINATOR


def simple_grid_bush_generator(wells):
    min_x, min_y = np.min(wells, axis=0)
    max_x, max_y = np.max(wells, axis=0)
    grid_step = get_grid_step(wells)
    x_num_points = ceil((max_x - min_x) / grid_step)
    y_num_points = ceil((max_y - min_y) / grid_step)
    bush_coords = np.array(
        list(product(np.linspace(min_x, max_x, x_num_points), np.linspace(min_y, max_y, y_num_points)))
    )

    return bush_coords


def simple_stochastic_bush_generator(wells):
    global RADIUS
    global PROPORTION
    global NUMBER_OF_SAMPLES
    sigma = RADIUS / 100

    bushes = []

    for well in wells:
        distribution = st.multivariate_normal(mean=well, cov=np.eye(len(well)) * sigma)
        curr_bushes = distribution.rvs(size=(NUMBER_OF_SAMPLES, 1)).tolist()
        bushes.extend(curr_bushes)

    bushes = np.array(bushes)
    convex_hull = Delaunay(wells)
    bushes = bushes[convex_hull.find_simplex(bushes) >= 0]
    indices = np.random.choice(list(range(len(bushes))), size=ceil(len(wells) * PROPORTION), replace=False)
    bushes = bushes[indices]

    return bushes


def generate_bushes_from_gm(means, covs, weights, total_number_of_points):
    dists_ids = np.random.choice(list(range(len(weights))), size=total_number_of_points, p=weights)
    bushes = []
    id_to_number_of_points = dict(Counter(dists_ids))

    for dist_id, number_of_points in id_to_number_of_points.items():
        curr_dist = st.multivariate_normal(mean=means[dist_id], cov=covs[dist_id])
        bushes.extend(curr_dist.rvs(size=number_of_points).tolist())

    return np.array(bushes)


def em_stochastic_bush_generator(wells):
    number_of_clusters = 1
    components_range = list(range(1, 8))
    bic = []

    for number_of_components in components_range:
        gmm = GaussianMixture(n_components=number_of_components)
        gmm.fit(wells)
        bic.append(gmm.bic(wells))

    number_of_clusters = max(number_of_clusters, components_range[np.argmin(bic)])
    gmm = GaussianMixture(n_components=number_of_clusters).fit(wells)

    global PROPORTION
    number_of_points = ceil(len(wells) * PROPORTION)
    bushes = generate_bushes_from_gm(gmm.means_, gmm.covariances_, gmm.weights_, number_of_points)
    convex_hull = Delaunay(wells)
    bushes = bushes[convex_hull.find_simplex(bushes) >= 0]

    return bushes


def intersect_with_wells(wells, bush_generator):
    bush_coords = bush_generator(wells)
    convex_hull = Delaunay(wells)
    bush_coords = bush_coords[convex_hull.find_simplex(bush_coords) >= 0]

    return bush_coords


def get_nearest_points_ids(wells_df, bush_coords):
    """
    in this method new bush coords are appending if it is needed
    :param wells_df:
    :param bush_coords:
    :return:
    """
    nearest_points = []
    wells = wells_df.loc[:, ['normalized_x', 'normalized_y']].values
    already_got = []

    for curr_bush in bush_coords:
        distances = np.array([dist(curr_bush, well) for well in wells])
        indices = np.argsort(distances)[:min(MAX_NUM_OF_WELLS_ON_BUSH, len(distances))]
        ids = wells_df.iloc[indices].id.values
        nearest_points.append(ids)
        already_got.extend(ids)

    already_got = set(already_got)
    not_taken = set(wells_df.id.values).difference(already_got)
    not_taken_points = wells_df[wells_df.id.isin(not_taken)].loc[:, ['normalized_x', 'normalized_y']].values

    for curr_bush in not_taken_points:
        distances = np.array([dist(curr_bush, well) for well in wells])
        indices = np.argsort(distances)[:min(MAX_NUM_OF_WELLS_ON_BUSH, len(distances))]
        ids = wells_df.iloc[indices].id.values
        nearest_points.append(ids)

    return nearest_points, np.append(bush_coords, not_taken_points, axis=0)


Bush = namedtuple('Bush', ['name', 'coords', 'cost', 'mip_variable'])
Well = namedtuple('Well', ['name', 'coords', 'cost', 'mip_variable'])


def get_cost_of_well(distance):
    global BASE_DRILL_COST
    global DRILL_METER_COST

    return DRILL_METER_COST * distance + BASE_DRILL_COST


def wrapper():
    global MAX_NUM_OF_WELLS_ON_BUSH
    a = [(*x, -1 if sum(x) == MAX_NUM_OF_WELLS_ON_BUSH else 1)
         for x in product((1, -1), repeat=MAX_NUM_OF_WELLS_ON_BUSH)]
    b = [-1 * Counter(x)[-1] + 0.1 for x in a]

    def _generate_wells_to_bush_constraint(bushes_wells, bush):
        variables = list(chain(map(lambda x: x.mip_variable, bushes_wells), [bush.mip_variable]))
        constraints = [xsum((a_ij * variable for a_ij, variable in zip(a_i, variables))) >= right
                       for a_i, right in zip(a, b)]

        return constraints

    return _generate_wells_to_bush_constraint


def get_all_wells_used_constraint(all_variables):
    wells_dict = defaultdict(list)

    for variable in all_variables:
        if isinstance(variable, Well):
            well_id = variable.name.split('_')[-1]
            wells_dict[well_id].append(variable)

    constraints = [xsum(map(lambda x: x.mip_variable, well_variables)) == 1
                   for well_variables in wells_dict.values()]
    return constraints


generate_wells_to_bush_constraint = wrapper()


def get_objective(all_variables):
    coefficients = map(lambda x: x.cost, all_variables)
    mip_variables = map(lambda x: x.mip_variable, all_variables)

    return minimize(xsum(coefficient * variable for coefficient, variable in zip(coefficients, mip_variables)))


def add_constraints(model, constraints):
    for constraint in constraints:
        model += constraint


def add_objective(model, objective):
    model.objective = objective


def optimize_bush_position(bush: Bush, wells: List[Well]):
    filtered_wells = list(filter(lambda well: well.mip_variable.x > 0, wells))

    def target(point):
        return np.sum([dist(point, well.coords) for well in filtered_wells])

    optimization_result = func_minimize(target, bush.coords)

    for coord_id, optimal_coord_value in enumerate(optimization_result.x):
        bush.coords[coord_id] = optimal_coord_value


def optimal_bushes_optimizer(all_variables):
    global MAX_NUM_OF_WELLS_ON_BUSH

    for bush_var_id in range(0, len(all_variables), MAX_NUM_OF_WELLS_ON_BUSH + 1):
        bush = all_variables[bush_var_id]

        if bush.mip_variable.x > 0:
            position = slice(bush_var_id + 1, bush_var_id + 1 + MAX_NUM_OF_WELLS_ON_BUSH)
            optimize_bush_position(bush, all_variables[position])


def mean_bush_position(bush: Bush, wells: List[Well]):
    filtered_wells = list(filter(lambda well: well.mip_variable.x > 0, wells))
    new_bush_coords = np.mean(list(map(lambda well: well.coords, filtered_wells)), axis=0)

    for coord_id, mean_coord_value in enumerate(new_bush_coords):
        bush.coords[coord_id] = mean_coord_value


def mean_bushes_optimizer(all_variables):
    info_dict = {}
    global MAX_NUM_OF_WELLS_ON_BUSH

    for bush_var_id in range(0, len(all_variables), MAX_NUM_OF_WELLS_ON_BUSH + 1):
        bush = all_variables[bush_var_id]

        if bush.mip_variable.x > 0:
            position = slice(bush_var_id + 1, bush_var_id + 1 + MAX_NUM_OF_WELLS_ON_BUSH)
            mean_bush_position(bush, all_variables[position])
            info_dict[bush.name.split('_')[1]] = list(map(lambda well: well.name.split('_')[-1],
                                                          filter(lambda well: well.mip_variable.x > 0,
                                                                 all_variables[position])))

    with open('optimizing.json', 'w') as destination:
        json.dump(info_dict, destination)


def base_worker_data_preparation(data, bush_coords, all_nearest_points_ids, model):
    all_variables = []
    all_constraints = []

    for bush_id, bush_x_y, nearest_points_ids in tqdm(zip(range(len(bush_coords)), bush_coords, all_nearest_points_ids),
                                                      total=len(bush_coords)):
        curr_bushes_wells = [None] * MAX_NUM_OF_WELLS_ON_BUSH
        bush_name = f'bush_{bush_id}'
        bush_variable = model.add_var(name=bush_name, var_type=BINARY)
        bush = Bush(name=bush_name,
                    coords=bush_x_y,
                    cost=BUSH_COST,
                    mip_variable=bush_variable)
        all_variables.append(bush)

        tmp_data = data[data.id.isin(nearest_points_ids)]
        nearest_points_ids = tmp_data.id.values
        nearest_points_coords = tmp_data.loc[:, ['normalized_x', 'normalized_y']].values

        for id_in_storage, point_id, point in zip(range(MAX_NUM_OF_WELLS_ON_BUSH),
                                                  nearest_points_ids, nearest_points_coords):
            distance = dist(bush_x_y, point)
            well_cost = get_cost_of_well(distance)
            well_name = bush_name + '_' + str(point_id)
            well_variable = model.add_var(name=well_name, var_type=BINARY)
            well = Well(name=well_name,
                        coords=point,
                        cost=well_cost,
                        mip_variable=well_variable)
            all_variables.append(well)
            curr_bushes_wells[id_in_storage] = well

        all_constraints.extend(generate_wells_to_bush_constraint(curr_bushes_wells, bush))

    all_constraints.extend(get_all_wells_used_constraint(all_variables))

    return all_variables, all_constraints


def base_worker(data, bush_generator, bush_optimizer, execution_time):
    wells = data.loc[:, ['normalized_x', 'normalized_y']].values
    bush_coords = intersect_with_wells(wells, bush_generator)

    global ITERATIONS_NUM
    global MAX_NUM_OF_WELLS_ON_BUSH

    for iteration_id in range(ITERATIONS_NUM):
        all_nearest_points_ids, bush_coords = get_nearest_points_ids(data, bush_coords)
        model = Model()
        all_variables, all_constraints = base_worker_data_preparation(data, bush_coords, all_nearest_points_ids, model)
        add_constraints(model, all_constraints)
        add_objective(model, get_objective(all_variables))
        model.threads = -1
        # TODO: comment max_solutions = 1
        # model.max_solutions = 1
        # 2 - optimal, 1 - feasible, 0 - balance (dafault option)
        # model.emphasis = 2
        status = model.optimize(execution_time)

        if status is not OptimizationStatus.NO_SOLUTION_FOUND:
            bush_optimizer(all_variables)
            bush_coords = np.array([all_variables[bush_var_id].coords
                                    for bush_var_id in range(0, len(all_variables), MAX_NUM_OF_WELLS_ON_BUSH + 1)])

    return status, all_variables


def read_data():
    data = pd.read_excel(PATH, engine='openpyxl')
    data.columns = ['id', 'layer', 'x', 'y', 'deposit']

    return data


def get_louvain_clusters(ids, points):
    graph = build_graph(ids, points, RADIUS, 1)
    communities = louvain(graph)

    return get_cluster_ids(ids, communities)


def get_default_clusters(ids, points):
    return [0] * len(ids)


BushPlotInfo = namedtuple('BushPlotInfo', ['id', 'coords', 'size'])


def get_bush_plot_info(bushes, bush_ids):
    bush_coords_dict = dict(zip(bush_ids, bushes))
    bush_sizes_dict = dict(Counter(bush_ids))
    global MAX_POINT_SIZE

    return [BushPlotInfo(bush_id, bush_coords_dict[bush_id], min(bush_sizes_dict[bush_id], MAX_POINT_SIZE))
            for bush_id in bush_sizes_dict.keys()]


def plot_decussing(data_name):
    # data preparation
    global PATH_TO_DATA_FOLDER
    data = pd.read_csv(os.path.join(PATH_TO_DATA_FOLDER, data_name + '.csv'))
    bushes = data.loc[:, ['bush_x', 'bush_y']].values
    wells = data.loc[:, ['normalized_x', 'normalized_y']].values
    bush_ids = data.bush_id.values

    # plotting edges
    edge_x = []
    edge_y = []

    for bush, well in zip(bushes, wells):
        edge_x.extend([bush[0], well[0], None])
        edge_y.extend([bush[1], well[1], None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    # plotting nodes
    bushes_plot_info = get_bush_plot_info(bushes, bush_ids)
    bushes = np.array([bush_plot_info.coords for bush_plot_info in bushes_plot_info])
    bush_size = [bush_plot_info.size for bush_plot_info in bushes_plot_info]
    bush_text = [bush_plot_info.id for bush_plot_info in bushes_plot_info]

    x = np.append(bushes[:, 0], wells[:, 0])
    y = np.append(bushes[:, 1], wells[:, 1])
    text = bush_text + data.id.values.tolist()
    size = bush_size + [1] * len(wells)
    color = ['bush'] * len(bushes) + ['well'] * len(wells)
    nodes = px.scatter(x=x, y=y, color_discrete_map={'bush': '#800000', 'well': '#FF8C00'},
                       color=color, size=size, text=text)
    result = go.Figure(data=[edge_trace, nodes.data[0], nodes.data[1]])

    # saving result
    global PATH_TO_PLOTS_FOLDER
    result.write_html(os.path.join(PATH_TO_PLOTS_FOLDER, data_name + '.html'))


def calculate_objective_value(variables: List[Union[Bush, Well]]):
    return np.sum([variable.cost * variable.mip_variable.x for variable in variables])


def one_bush_multiple_wells_test(variables: List[Union[Bush, Well]]):
    wells = list(filter(lambda x: isinstance(x, Well) and x.mip_variable.x > 0, variables))
    result = len(np.unique([well.name.split('_')[-1] for well in wells])) == len(wells)
    msg = f'{bcolors.OKGREEN if result else bcolors.FAIL}one_bush_multiple_wells: {result}{bcolors.ENDC}'
    print(msg)
    logging.info(msg)


def is_bushing_valid_test(variables: List[Union[Bush, Well]]):
    bushes = list(filter(lambda x: isinstance(x, Bush) and x.mip_variable.x > 0, variables))
    wells = list(filter(lambda x: isinstance(x, Well) and x.mip_variable.x > 0, variables))

    used_bushes = set([bush.name.split('_')[1] for bush in bushes])
    bushes_used_by_wells = set([well.name.split('_')[1] for well in wells])
    result = len(used_bushes.intersection(bushes_used_by_wells)) == len(used_bushes)
    msg = f'{bcolors.OKGREEN if result else bcolors.FAIL}is_bushing_valid: {result}{bcolors.ENDC}'
    print(msg)
    logging.info(msg)


def clusterization_worker(data, get_clusters, bush_generator, bush_optimizer, execution_time):
    points = data.loc[:, ['normalized_x', 'normalized_y']].values
    data['cluster_id'] = get_clusters(data.id.values, points)
    clusters = set(data.cluster_id.values)
    global max_clusters
    max_clusters = max(max_clusters, len(clusters))
    all_variables = []
    statuses = []

    for cluster_id in clusters:
        wells_df = data[data.cluster_id == cluster_id]
        status, current_all_variables = base_worker(wells_df, bush_generator, bush_optimizer, execution_time)
        all_variables.extend(current_all_variables)
        statuses.append(status)

    status = OptimizationStatus.NO_SOLUTION_FOUND if OptimizationStatus.NO_SOLUTION_FOUND in statuses else statuses[0]
    return status, all_variables


def test_solution(all_variables):
    print()

    if all_variables[0].mip_variable.x is not None:
        msg = f'{bcolors.OKCYAN}Solution found with metric value - ' \
              f'{calculate_objective_value(all_variables)}{bcolors.ENDC}'
        print(msg)
        logging.info(msg)
        one_bush_multiple_wells_test(all_variables)
        is_bushing_valid_test(all_variables)
    else:
        print(f"{bcolors.OKCYAN}No solution found{bcolors.ENDC}", end='\n\n')


def save_data(data: pd.DataFrame, all_variables, name):
    bush_coords_dict = defaultdict(np.array)
    bush_ids_dict = defaultdict(str)

    for bush_var_id in range(0, len(all_variables), MAX_NUM_OF_WELLS_ON_BUSH + 1):
        bush_id = all_variables[bush_var_id].name.split('_')[1]
        current_bush_coords = all_variables[bush_var_id].coords

        for well_var_id in range(1, MAX_NUM_OF_WELLS_ON_BUSH + 1):
            well_var = all_variables[well_var_id + bush_var_id]

            if well_var.mip_variable.x > 0:
                well_id = well_var.name.split('_')[-1]
                bush_coords_dict[well_id] = current_bush_coords
                bush_ids_dict[well_id] = bush_id

    bush_coords = np.array([bush_coords_dict[str(well_id)] for well_id in data.id.values])
    bush_ids = [bush_ids_dict[str(well_id)] for well_id in data.id.values]
    data['bush_id'] = bush_ids
    data['bush_x'] = bush_coords[:, 0]
    data['bush_y'] = bush_coords[:, 1]
    global PATH_TO_DATA_FOLDER
    data.to_csv(os.path.join(PATH_TO_DATA_FOLDER, name + '.csv'))


def is_solution_found(all_varibales):
    return all_varibales[0].mip_variable.x is not None


def run(data, clusterizer, grid_generator, bush_optimizer, execution_time):
    """
    main pipeline
    :param data: have to be deep copied
    :param clusterizer:
    :param grid_generator:
    :param bush_optimizer:
    :param execution_time:
    :return:
    """
    global MAX_NUM_OF_WELLS_ON_BUSH
    name = '_'.join(chain(map(lambda function: function.__name__, [clusterizer, grid_generator, bush_optimizer]),
                          [str(MAX_NUM_OF_WELLS_ON_BUSH)]))
    print(f"{bcolors.OKGREEN}{name}{bcolors.ENDC}")
    status, all_variables = clusterization_worker(data, clusterizer, grid_generator, bush_optimizer, execution_time)

    if status is not OptimizationStatus.NO_SOLUTION_FOUND:
        logging.info(name)
        test_solution(all_variables)
        save_data(data, all_variables, name)
        plot_decussing(name)
    print(end='\n\n\n')


def main():
    # read data
    data = read_data()

    # scale data
    scaler = MyMinMaxScaler()
    points = scaler.fit_transform(data.loc[:, ['x', 'y']].values)
    global RADIUS
    RADIUS = scaler.reset_radius(RADIUS)
    data['normalized_x'] = points[:, 0]
    data['normalized_y'] = points[:, 1]

    # main work
    global MAX_TIME
    run(data.copy(deep=True), get_louvain_clusters, simple_grid_bush_generator, optimal_bushes_optimizer, MAX_TIME)
    # run(data.copy(deep=True), get_default_clusters, simple_stochastic_bush_generator, optimal_bushes_optimizer, MAX_TIME)
    # run(data.copy(deep=True), get_default_clusters, em_stochastic_bush_generator, optimal_bushes_optimizer, MAX_TIME)


main()
