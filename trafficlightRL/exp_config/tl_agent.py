"""Traffic Light Agent"""

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams
from flow.envs import TrafficLightGridPOEnvTest
from flow.networks import Network

import os

HORIZON = 500
N_ROLLOUTS = 20 #*3
N_CPUS = 2 #*2

intersection_dir = "/home/ubuntu/flow/"

net_params = NetParams(
    template={
      "net": os.path.join(intersection_dir, "trafficLightRL/network_sa.net.xml"),
      "vtype": os.path.join(intersection_dir, "trafficLightRL/vtypeflow_sa.add.xml"),
      "rou": os.path.join(intersection_dir, "trafficLightRL/routes_sa.rou.xml")
    }
)

additional_env_params = {
    'target_velocity': 50,
    'switch_time':3.5,
    'num_observed': 2,
    'discrete': False,
    'tl_type': 'controlled'
}

# create the remainding parameters
env_params = EnvParams(horizon=HORIZON, additional_params=additional_env_params)
sim_params = SumoParams(render=False, sim_step=1, restart_instance=True)
vehicles = VehicleParams()

# the above variable is added to initial_config
initial_config = InitialConfig(
    spacing="random",
    lanes_distribution=float('inf'), 
    shuffle=True
)

#*1
flow_params = dict(
    exp_tag='train_simulation_area',
    env_name=TrafficLightGridPOEnvTest, #*4
    network=Network,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)
