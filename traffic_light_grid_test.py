"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
"""

import numpy as np
import re

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

from flow.core import rewards
from flow.envs.base import Env

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
}

ADDITIONAL_PO_ENV_PARAMS = {
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 2,
    # velocity to use in reward functions
    "target_velocity": 50,
}


class TrafficLightGridEnvTest(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.rows = 7
        self.cols = 1
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')
        TLagents = ['tl0','tl1', 'tl2', 'tl3', 'tl4', 'tl5', 'tl6']

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        # Keeps track of the last time the traffic lights in an intersection were allowed to change (the last time the lights were allowed to change from a red-green state to a red-yellow state.)
        self.last_change = np.zeros((self.rows * self.cols, 1))
        # Keeps track of the direction of the intersection (the direction that is currently being allowed to flow. 0 indicates flow from top to bottom, and 1 indicates flow from left to right.)
        self.direction = np.zeros((self.rows * self.cols, 1))
        # Value of 1 indicates that the intersection is in a red-yellow state. value 0 indicates that the intersection is in a red-green state.
        self.currently_yellow = np.zeros((self.rows * self.cols, 1))
        # when this hits min_switch_time we change from yellow to red
        # the second column indicates the direction that is currently being allowed to flow. 0 is flowing top to bottom, 1 is left to right
        # For third column, 0 signifies yellow and 1 green or red
        self.min_switch_time = env_params.additional_params["switch_time"]

        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                if 'tl0' in TLagents[i]:
                    self.k.traffic_light.set_state(
                        node_id='tl' + str(i), state="GGrrr")
                if 'tl1' in TLagents[i]:
                    self.k.traffic_light.set_state(
                        node_id='tl' + str(i), state="rrrGG")
                if 'tl2' in TLagents[i]:
                    self.k.traffic_light.set_state(
                        node_id='tl' + str(i), state="rrrGGG")
                if 'tl3' in TLagents[i]:
                    self.k.traffic_light.set_state(
                        node_id='tl' + str(i), state="GGrGrG")
                if 'tl4' in TLagents[i]:
                    self.k.traffic_light.set_state(
                        node_id='tl' + str(i), state="GGrrrG")
                if 'tl5' in TLagents[i]:
                    self.k.traffic_light.set_state(
                        node_id='tl' + str(i), state="Grrr")
                if 'tl6' in TLagents[i]:
                    self.k.traffic_light.set_state(
                        node_id='tl' + str(i), state="GGrrrG")
                self.currently_yellow[i] = 0

        # # Additional Information for Plotting
        # self.edge_mapping = {"top": [], "bot": [], "right": [], "left": []}
        # for i, veh_id in enumerate(self.k.vehicle.get_ids()):
        #     edge = self.k.vehicle.get_edge(veh_id)
        #     for key in self.edge_mapping:
        #         if key in edge:
        #             self.edge_mapping[key].append(i)
        #             break

        # check whether the action space is meant to be discrete or continuous
        self.discrete = env_params.additional_params.get("discrete", False)

    @property
    def action_space(self):
        if self.discrete:
            return Discrete(2 ** self.num_traffic_lights)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(self.num_traffic_lights,),
                dtype=np.float32)

    @property
    def observation_space(self):
        speed = Box(
            low=0,
            high=1,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        dist_to_intersec = Box(
            low=0.,
            high=np.inf,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        edge_num = Box(
            low=0.,
            high=1,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        traffic_lights = Box(
            low=0.,
            high=1,
            shape=(3 * self.rows * self.cols,),
            dtype=np.float32)
        return Tuple((speed, dist_to_intersec, edge_num, traffic_lights))

    def get_state(self):
        # compute the normalizers
        max_dist = max(341.15,17.43,1687.44)

        # get the state arrays
        speeds = [
            self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
            for veh_id in self.k.vehicle.get_ids()
        ]
        dist_to_intersec = [
            self.get_distance_to_intersection(veh_id) / max_dist
            for veh_id in self.k.vehicle.get_ids()
        ]
        edges = [
            self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
            (self.k.network.network.num_edges - 1)
            for veh_id in self.k.vehicle.get_ids()
        ]

        state = [
            speeds, dist_to_intersec, edges,
            self.last_change.flatten().tolist(),
            self.direction.flatten().tolist(),
            self.currently_yellow.flatten().tolist()
        ]
        return np.array(state)

    def _apply_rl_actions(self, rl_actions):
        TLagents = ['tl0','tl1','tl2','tl3','tl4','tl5','tl6']
        # check if the action space is discrete
        if self.discrete:
            # convert single value to list of 0's and 1's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            rl_mask = rl_actions > 0.0
        
        for i, action in enumerate(rl_mask):
            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:#di0
                        #RG
                        if 'tl0' in TLagents[0]:
                            self.k.traffic_light.set_state(
                                node_id='tl0', state="GGrrr")
                        if 'tl1' in TLagents[1]:
                            self.k.traffic_light.set_state(
                                node_id='tl1', state="rrrGG")
                        if 'tl2' in TLagents[2]:
                            self.k.traffic_light.set_state(
                                node_id='tl2', state="rrrGGG")
                        if 'tl3' in TLagents[3]:
                            self.k.traffic_light.set_state(
                                node_id='tl3', state="GGrGrG")
                        if 'tl4' in TLagents[4]:
                            self.k.traffic_light.set_state(
                                node_id='tl4', state="GGrrrG")
                        if 'tl5' in TLagents[5]:
                            self.k.traffic_light.set_state(
                                node_id='tl5', state="Grrr")
                        if 'tl6' in TLagents[6]:
                            self.k.traffic_light.set_state(
                                node_id='tl6', state="GGrrG")      
                    elif self.direction[i] == 1: #dir1
                        if 'tl0' in TLagents[0]:
                            self.k.traffic_light.set_state(
                                node_id='tl0', state="rGGGr")
                        if 'tl1' in TLagents[1]:
                            self.k.traffic_light.set_state(
                                node_id='tl1', state="GGrrG")
                        if 'tl2' in TLagents[2]:
                            self.k.traffic_light.set_state(
                                node_id='tl2', state="GGrrrG")
                        if 'tl3' in TLagents[3]:
                            self.k.traffic_light.set_state(
                                node_id='tl3', state="rGGGrG")
                        if 'tl4' in TLagents[4]:
                            self.k.traffic_light.set_state(
                                node_id='tl4', state="rrGrrG")
                        if 'tl5' in TLagents[5]:
                            self.k.traffic_light.set_state(
                                node_id='tl5', state="rGrr")
                        if 'tl6' in TLagents[6]:
                            self.k.traffic_light.set_state(
                                node_id='tl6', state="rGGrr")   
                    elif self.direction[i] == 2:#dir2
                        if 'tl0' in TLagents[0]:
                            self.k.traffic_light.set_state(
                                node_id='tl0', state="rGrGG")
                        if 'tl1' in TLagents[1]:
                            self.k.traffic_light.set_state(
                                node_id='tl1', state="rGGrG")
                        if 'tl2' in TLagents[2]:
                            self.k.traffic_light.set_state(
                                node_id='tl2', state="rGGGrr")
                        if 'tl3' in TLagents[3]:
                            self.k.traffic_light.set_state(
                                node_id='tl3', state="rrrGGG")
                        if 'tl4' in TLagents[4]:
                            self.k.traffic_light.set_state(
                                node_id='tl4', state="rGrGrr")
                        if 'tl5' in TLagents[5]:
                            self.k.traffic_light.set_state(
                                node_id='tl5', state="rrGr")
                        if 'tl6' in TLagents[6]:
                            self.k.traffic_light.set_state(
                                node_id='tl6', state="rrrGG")
                    else:
                        if 'tl0' in TLagents[0]:
                            self.k.traffic_light.set_state(
                                node_id='tl0', state="rGGGr")
                        if 'tl1' in TLagents[1]:
                            self.k.traffic_light.set_state(
                                node_id='tl1', state="GGrrG")
                        if 'tl2' in TLagents[2]:
                            self.k.traffic_light.set_state(
                                node_id='tl2', state="GGrrrG")
                        if 'tl3' in TLagents[3]:
                            self.k.traffic_light.set_state(
                                node_id='tl3', state="rGGGrG")
                        if 'tl4' in TLagents[4]:
                            self.k.traffic_light.set_state(
                                node_id='tl4', state="rrrrGG")
                        if 'tl5' in TLagents[5]:
                            self.k.traffic_light.set_state(
                                node_id='tl5', state="rrrG")
                        if 'tl6' in TLagents[6]:
                            self.k.traffic_light.set_state(
                               node_id='tl6', state="rGGrr")       
                    self.currently_yellow[i] = 0
            else: #yellow phase
                if action:
                    if self.direction[i] == 0: #dir 0
                        if TLagents[0] == "tl0":
                            self.k.traffic_light.set_state(
                                node_id='tl0', state="yGrrr")
                        if TLagents[1] == "tl1":
                            self.k.traffic_light.set_state(
                                node_id='tl1', state="rrryG")
                        if TLagents[2] == "tl2":
                            self.k.traffic_light.set_state(
                                node_id='tl2', state="rrryyG")
                        if TLagents[3] == "tl3":
                            self.k.traffic_light.set_state(
                                node_id='tl3', state="yGrGrG")
                        if TLagents[4] == "tl4":
                            self.k.traffic_light.set_state(
                                node_id='tl4', state="yyrrrG")
                        if TLagents[5] == "tl5":
                            self.k.traffic_light.set_state(
                                node_id='tl5', state="yrrr")
                        if TLagents[6] == "tl6":
                            self.k.traffic_light.set_state(
                                node_id='tl6', state="yGrry")   
                    elif self.direction[i] == 1:#dir 1
                        if TLagents[0] == "tl0":
                            self.k.traffic_light.set_state(
                                node_id='tl0', state="rGyGr")
                        if TLagents[1] == "tl1":
                            self.k.traffic_light.set_state(
                                node_id='tl1', state="yGrrG")
                        if TLagents[2] == "tl2":
                            self.k.traffic_light.set_state(
                                node_id='tl2', state="yGrrry")
                        if TLagents[3] == "tl3":
                            self.k.traffic_light.set_state(
                                node_id='tl3', state="ryyGrG")
                        if TLagents[4] == "tl4":
                            self.k.traffic_light.set_state(
                                node_id='tl4', state="rryrry")
                        if TLagents[5] == "tl5":
                            self.k.traffic_light.set_state(
                                node_id='tl5', state="ryrr")
                        if TLagents[6] == "tl6":
                            self.k.traffic_light.set_state(
                                node_id='tl6', state="ryyrr")  
                    elif self.direction[i] == 2:#dir2
                        if TLagents[0] == "tl0":
                            self.k.traffic_light.set_state(
                                node_id='tl0', state="rGryy")
                        if TLagents[1] == "tl1":
                            self.k.traffic_light.set_state(
                                node_id='tl1', state="ryyrG")
                        if TLagents[2] == "tl2":
                             self.k.traffic_light.set_state(
                                 node_id='tl2', state="ryyGrr")
                        if TLagents[3] == "tl3":
                             self.k.traffic_light.set_state(
                                 node_id='tl3', state="rrrGyG")
                        if TLagents[4] == "tl4":
                            self.k.traffic_light.set_state(
                                node_id='tl4', state="ryryrr")
                        if TLagents[5] == "tl5":
                            self.k.traffic_light.set_state(
                                node_id='tl5', state="rryr")
                        if TLagents[6] == "tl6":
                            self.k.traffic_light.set_state(
                                node_id='tl6', state="rrryG")
                    else:
                        if 'tl0' in TLagents[0]:
                            self.k.traffic_light.set_state(
                                node_id='tl0', state="rGyGr")
                        if 'tl1' in TLagents[1]:
                            self.k.traffic_light.set_state(
                                node_id='tl1', state="yGrrG")
                        if 'tl2' in TLagents[2]:
                            self.k.traffic_light.set_state(
                                node_id='tl2', state="yGrrry")
                        if 'tl3' in TLagents[3]:
                            self.k.traffic_light.set_state(
                                node_id='tl3', state="ryyGrG")
                        if 'tl4' in TLagents[4]:
                            self.k.traffic_light.set_state(
                                node_id='tl4', state="rrrryG")
                        if 'tl5' in TLagents[5]:
                            self.k.traffic_light.set_state(
                                node_id='tl5', state="rrry")
                        if 'tl6' in TLagents[6]:
                            self.k.traffic_light.set_state(
                                node_id='tl6', state="ryyrr")          
                    self.last_change[i] = 0.0
                    if i+1 < 7:
                        if self.direction[i] == 0:
                            self.direction[i+1] = 1
                        elif self.direction[i] == 1:
                            self.direction[i+1] = 2
                        elif self.direction[i] == 2:
                            self.direction[i+1] = 3
                        else:
                            self.direction[i+1] = 0
                    self.currently_yellow[i] = 1

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return - rewards.min_delay_unscaled(self) \
            - rewards.boolean_action_penalty(rl_actions >= 0.5, gain=1.0)

    # ===============================
    # ============ UTILS ============
    # ===============================

    def get_distance_to_intersection(self, veh_ids):
        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        edge_id = self.k.vehicle.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'tl' in edge_id:
            return 0
        edge_len = self.k.network.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def _convert_edge(self, edges):
        if isinstance(edges, list):
            return [self._split_edge(edge) for edge in edges]
        else:
            return self._split_edge(edges)

    def _split_edge(self, edge):
        """Act as utility function for convert_edge."""
        if edge:
            if edge[0] == ":":  # center
                center_index = int(edge.split("tl")[1][0])
                base = ((self.cols + 1) * self.rows * 2) \
                    + ((self.rows + 1) * self.cols * 2)
                return base + center_index + 1
            else:
                pattern = re.compile(r"[a-zA-Z]+")
                edge_type = pattern.match(edge).group()
                edge = edge.split(edge_type)[1].split('_')
                row_index, col_index = [int(x) for x in edge]
                if edge_type in ['bot', 'top']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * (row_index + 1))
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'bot' else edge_num + 1
                if edge_type in ['left', 'right']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * row_index)
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'left' else edge_num + 1
        else:
            return 0

    def _get_relative_node(self, agent_id, direction):
        ID_IDX = 1
        agent_id_num = int(agent_id.split("tl")[ID_IDX])
        if direction == "top":
            node = agent_id_num + self.cols
            if node >= self.cols * self.rows:
                node = -1
        elif direction == "bottom":
            node = agent_id_num - self.cols
            if node < 0:
                node = -1
        elif direction == "left":
            if agent_id_num % self.cols == 0:
                node = -1
            else:
                node = agent_id_num - 1
        elif direction == "right":
            if agent_id_num % self.cols == self.cols - 1:
                node = -1
            else:
                node = agent_id_num + 1
        else:
            raise NotImplementedError

        return node

    def additional_command(self):
        for veh_id in self.k.vehicle.get_ids():
            self._reroute_if_final_edge(veh_id)

    def _reroute_if_final_edge(self, veh_id):
        edge = self.k.vehicle.get_edge(veh_id)
        if edge == "":
            return
        if edge[0] == ":":  # center edge
            return
        pattern = re.compile(r"[a-zA-Z]+")
        edge_type = pattern.match(edge).group()
        edge = edge.split(edge_type)[1].split('_')
        row_index, col_index = [int(x) for x in edge]

        # find the route that we're going to place the vehicle on if we are going to remove it
        route_id = None
        if edge_type == 'bot' and col_index == self.cols:
            route_id = "bot{}_0".format(row_index)
        elif edge_type == 'top' and col_index == 0:
            route_id = "top{}_{}".format(row_index, self.cols)
        elif edge_type == 'left' and row_index == 0:
            route_id = "left{}_{}".format(self.rows, col_index)
        elif edge_type == 'right' and row_index == self.rows:
            route_id = "right0_{}".format(col_index)

        if route_id is not None:
            type_id = self.k.vehicle.get_type(veh_id)
            lane_index = self.k.vehicle.get_lane(veh_id)
            # remove the vehicle
            self.k.vehicle.remove(veh_id)
            # reintroduce it at the start of the network
            self.k.vehicle.add(
                veh_id=veh_id,
                edge=route_id,
                type_id=str(type_id),
                lane=str(lane_index),
                pos="0",
                speed="max")

    def get_closest_to_intersection(self, edges, num_closest, padding=False):
        if num_closest <= 0:
            raise ValueError("Function get_closest_to_intersection called with parameter num_closest={}, but num_closest should be positive".format(num_closest))

        if isinstance(edges, list):
            ids = [self.get_closest_to_intersection(edge, num_closest)
                   for edge in edges]
            # flatten the list and return it
            return [veh_id for sublist in ids for veh_id in sublist]

        # get the ids of all the vehicles on the edge 'edges' ordered by increasing distance to end of edge (intersection)
        veh_ids_ordered = sorted(self.k.vehicle.get_ids_by_edge(edges),
                                 key=self.get_distance_to_intersection)

        # return the ids of the num_closest vehicles closest to the
        # intersection, potentially with ""-padding.
        pad_lst = [""] * (num_closest - len(veh_ids_ordered))
        return veh_ids_ordered[:num_closest] + (pad_lst if padding else [])


class TrafficLightGridPOEnvTest(TrafficLightGridEnvTest):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_PO_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of vehicles nearest each intersection that is observed in the state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 2)

        # used during visualization
        self.observed_ids = []

    @property
    def observation_space(self):
        tl_box = Box(
            low=0.,
            high=1000,
            shape=(3 * 4 * self.num_observed * self.num_traffic_lights +
                   2 * len(self.k.network.get_edge_list()) +
                   3 * self.num_traffic_lights,),
            dtype=np.float32)
        return tl_box

    def get_state(self):
        speeds = []
        dist_to_intersec = []
        edge_number = []
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        max_dist = max(341.15,17.43,1687.44)
        all_observed_ids = []

        for _, edges in self.network.node_mapping:
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection(edge, self.num_observed)
                all_observed_ids += observed_ids

                # check which edges we have so we can always pad in the right positions
                speeds += [
                    self.k.vehicle.get_speed(veh_id) / max_speed
                    for veh_id in observed_ids
                ]
                dist_to_intersec += [
                    (self.k.network.edge_length(
                        self.k.vehicle.get_edge(veh_id)) -
                        self.k.vehicle.get_position(veh_id)) / max_dist
                    for veh_id in observed_ids
                ]
                edge_number += \
                    [self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
                     (self.k.network.network.num_edges - 1)
                     for veh_id in observed_ids]

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    speeds += [0] * diff
                    dist_to_intersec += [0] * diff
                    edge_number += [0] * diff

        # now add in the density and average velocity on the edges
        density = []
        velocity_avg = []
        for edge in self.k.network.get_edge_list():
            ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(ids) > 0:
                vehicle_length = 5
                density += [vehicle_length * len(ids) /
                            self.k.network.edge_length(edge)]
                velocity_avg += [np.mean(
                    [self.k.vehicle.get_speed(veh_id) for veh_id in
                     ids]) / max_speed]
            else:
                density += [0]
                velocity_avg += [0]
        self.observed_ids = all_observed_ids
        return np.array(
            np.concatenate([
                speeds, dist_to_intersec, edge_number, density, velocity_avg,
                self.last_change.flatten().tolist(),
                self.direction.flatten().tolist(),
                self.currently_yellow.flatten().tolist()
            ]))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return (- rewards.min_delay_unscaled(self) +
                    rewards.penalize_standstill(self, gain=0.2))

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        [self.k.vehicle.set_observed(veh_id) for veh_id in self.observed_ids]


class TrafficLightGridBenchmarkEnvTest(TrafficLightGridPOEnvTest):
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return rewards.desired_velocity(self)


class TrafficLightGridTestEnvTest(TrafficLightGridEnvTest):
    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """No return, for testing purposes."""
        return 0
