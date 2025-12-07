import os
import time
import random

import tqdm
import networkx as nx

import numpy as np

from .sdf import *
from .collision_detection import *
from .distance import *
from .extend import *
from .rotation_utils import *
from .sample import *
from .visualize import *
from .nearest_neighbors import *

def argmin(function, sequence):
    values = list(sequence)
    scores = [function(x) for x in values]
    return values[scores.index(min(scores))]


def update_visits(graph, node):
    """
    Updates the number of visits in a specific node.

    Args:
        graph (networkx.Graph): The graph object.
        node: The node to update the visits for.

    Returns:
        None
    """
    if node in graph:
        if 'visits' in graph.nodes[node]:
            graph.nodes[node]['visits'] += 1
        else:
            graph.nodes[node]['visits'] = 1

# def choose_less_visited_node(graph):
#     """
#     Chooses the less visited node from the given graph.
#
#     Args:
#         graph (networkx.Graph): The graph object.
#
#     Returns:
#         The less visited node.
#     """
#     less_visited_node = None
#     min_visits = float('inf')
#
#     for node in graph.nodes:
#         visits = graph.nodes[node].get('visits', 0)
#         if visits < min_visits:
#             min_visits = visits
#             less_visited_node = node
#
#     return less_visited_node


def choose_less_visited_node(graph, power=2):
    """
    Chooses a random node with a strong bias towards less visited nodes from the given graph.

    Args:
        graph (networkx.Graph): The graph object.
        power (float): The power parameter for bias. Higher values increase the bias. Defaults to 2.

    Returns:
        The randomly selected node with bias towards less visited nodes.
    """
    max_visits = max(graph.nodes[node].get('visits', 0) for node in graph.nodes)
    probabilities = [(1.01 - graph.nodes[node].get('visits', 0) / max_visits) ** power for node in graph.nodes]
    chosen_node = random.choices(list(graph.nodes), weights=probabilities)[0]
    return chosen_node


def rotate_config(q, q_rotation):
    return SE3_2_xyzrpy( xyzrpy_2_SE3(q_rotation) * xyzrpy_2_SE3(q))


# rotate the entire system (m1, root_start tree) every N_ROTATE iterations
def random_rotate_system(q1, q2_start, q2_end, graph, nn, bv, sampler=None):
    start = time.time()
    bv.removeAllUserDebugItems()

    system_rotation = [0, 0, 0, 0, 0, 0]
    system_rotation[3:] += np.random.rand(3)
    q1 = rotate_config(q1, system_rotation)[:]
    q2_start = rotate_config(q2_start, system_rotation)[:]
    q2_end = rotate_config(q2_end, system_rotation)[:]
    bv.set_object_configuration("m1", xyzrpy_2_SE3(q1))
    new_graph = nx.Graph()
    nodes_dict = dict()
    new_nn = NearestNeighborsCached(NearestNeighbors_sklearn(metric=Metric_Euclidean))
    for node in graph.nodes.items():
        nodes_dict[node[0]] = tuple(rotate_config(node[0], tuple(system_rotation)))

    for sample_point in nn.cache:
        new_nn.add_point(rotate_config(sample_point, tuple(system_rotation)))

    if not sampler is None:
        for i in range(len(sampler.points)):
            sampler.points[i] = tuple(rotate_config(sampler.points[i], system_rotation)[:])

    jump_size = int(len(graph.edges.items()) / 10000) + 1
    count = 0
    for edge in graph.edges.items():
        count += 1
        q_from, q_to = nodes_dict[edge[0][0]], nodes_dict[edge[0][1]]
        new_graph.add_edge(q_from, q_to)
        # draw only up to 10,000 edges as drawing lines is time consuming
        if count % jump_size != 0:
            continue
        # bv.add_debug_line_from_xyz(q_from[:3], q_to[:3], [1, 0, 0])

    print('rotate {} took {:.2f} sec'.format(system_rotation, time.time()-start))

    return q1, q2_start, q2_end, new_graph, new_nn


class RRT(object):
    def __init__(self, m1: SDFMesh, q1: list, m2: SDFMesh, q2_start, q2_end, device, bv):
        """
        Mesh1 is static, Mesh2 is dynamic.
        """
        self.m1 = m1
        self.q1 = q1
        self.m2 = m2
        self.q2_start = q2_start
        self.q2_end = q2_end
        self.bv = bv
        self.device = device

        self.verbose = True
        
        self.distance = distance_between_configurations

        self.sample_position_scale = (m1.max_norm + m2.max_norm) * 2
        self.sample_rotation_scale = 1.0
        # self.sampler = BiasedTowardsTightSampelr()
        self.sampler = UnionSampler()
        # self.sampler = TightSampler()
        self.sample = lambda: self.sampler.sample(self.sample_position_scale, self.sample_rotation_scale)
        # self.sample = lambda: union_sample(position_scale=self.sample_position_scale, rotation_scale=self.sample_rotation_scale)

        self.log_path = None
        self.tree_path = None

        self.extend_eta = 0.1
        self.extend_threshold = -0.01 
        self.epsilon_contact = 0.25 * self.extend_eta
        self.extend = None # Make sure to add extend implementation in main!

        self.update_visualization = update_rrt_visualization

        self.is_point_valid = lambda q: not is_collision(
            self.m1, self.q1, self.m2, q, self.extend_threshold, self.device
        )
        self.is_edge_valid = lambda q_s, q_e: is_edge_valid(
            self.m1, self.q1, self.m2, q_s, q_e, 
            int(distance_between_configurations(q_s, q_e) / self.extend_eta),
            self.extend_threshold, self.device
        )
    
    def plan(self, num_iterations=int(1e6), timeout=1e6):
        """
        RRT Planner
        """

        # if not self.is_point_valid(q_start) or not self.is_point_valid(q_end):
        #     print('start or end points are in collision!')
        #     return None

        self.sampler.aabb.update(*self.q2_start[:3])
        nn = NearestNeighborsCached(NearestNeighbors_sklearn(metric=Metric_Euclidean))

        if os.path.isfile(self.tree_path) and False:
            # continue from previous run
            with open(self.tree_path, 'rb') as fp:
                root_start = pickle.load(fp)
            for configuration in tqdm.tqdm(root_start):
                nn.add_point(configuration)
        else:
            root_start = nx.Graph()
            root_start.add_node(tuple(fix_configuration(self.q2_start)))
            # update_visits(root_start, tuple(fix_configuration(q_start)))
            nn.add_point(self.q2_start)

        # Helper variable holding the last value
        last1 = None

        for iteration in range(num_iterations):
            n_rotate = 100   # minimal number of rotations between rotations. later increase to 10%
            ###################################################
            if iteration % n_rotate == 1:
                self.q1, self.q2_start, self.q2_end, root_start, nn =\
                    random_rotate_system(self.q1, self.q2_start,  self.q2_end, root_start, nn, self.bv)
            if n_rotate * 10 < iteration:
                n_rotate *= 10
            ###################################################
            self.iteration = iteration
            if self.bv is not None:
                self.bv.step()
            if self.verbose:
                print("simple rrt - Iteration #{}".format(iteration))

            # if iteration >= 10:
            if iteration >= 15000:
                print("TIMEOUT")
                return None

            s = self.sample()

            # last1 = argmin(lambda n: self.distance(n, s), root_start)
            last1 = nn.k_nearest(np.array(s), k=1)[0]
            if np.random.random() < 0.9:
                direction = get_unit_direction(last1, s)
            else:
                direction = union_sample()

            #################
            # Extend
            #################
            extend_start = time.time()
            qs, _ = self.extend(last1, direction)
            extend_time = time.time() - extend_start

            # log_line = f'{iteration},\t{extend_time}\n'
            log_line = 'iteration: {}, time: {:.2f} [sec]'.format(iteration, extend_time)
            print(log_line)
            if self.log_path is not None:
                with open(self.log_path, 'a') as fp:
                    fp.write(log_line)

            for _, q in enumerate(qs):
                root_start.add_edge(tuple(fix_configuration(q)), tuple(last1))
                last1 = q

            if len(qs) > 0:
                if self.bv is not None:
                    self.update_visualization(self.bv, qs[0], qs[-1], [1, 0, 0], update=False)
                self.sampler.aabb.update(*qs[-1][:3])
                for q in qs:
                    nn.add_point(q)

                if self.is_edge_valid(last1, self.q2_end):
                    print("found path", last1, self.q2_end)
                    path1 = nx.algorithms.shortest_path(root_start, tuple(self.q2_start), tuple(last1))
                    path_12 = pave_short_edge(last1, self.q2_end, 30)
                    if self.verbose:
                        print("Found path after {} iterations".format(iteration))
                    return path1 + path_12
                
            if (iteration % 100 == 0) and self.tree_path is not None:
                with open(self.tree_path, 'wb') as fp:
                    pickle.dump(root_start, fp)
        print("RRT: reached max iterations without finding a path")
        return None

