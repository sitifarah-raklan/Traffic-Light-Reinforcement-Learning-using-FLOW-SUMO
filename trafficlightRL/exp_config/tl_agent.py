"""Traffic Light Agent"""

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams
from flow.envs import TrafficLightGridPOEnvTest
from flow.envs import CustomTrafficLightPOEnv
from flow.networks import Network

import os

HORIZON = 900
N_ROLLOUTS = 10 
N_CPUS = 2 

intersection_dir = "/home/ubuntu/flow/playground/Simulation_Area"

net_params = NetParams(
    template={
      "net": os.path.join(intersection_dir, "update/updated_network.net.xml"),
      "vtype": os.path.join(intersection_dir, "vtypeflow_sa.add.xml"),
      "rou": os.path.join(intersection_dir, "update/updated_route.rou.xml")
    }
)

additional_env_params = {
    'target_velocity': 50,
    'switch_time':35.0,
    'yellow_time': 5.0,
    'num_observed': 5,
    'discrete': False,
    'tl_type': 'actuated'
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
    exp_tag='new_training2',
    env_name=CustomTrafficLightPOEnv,
    network=Network,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)
