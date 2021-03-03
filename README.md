# Traffic-Light-Reinforcement-Learning-using-FLOW-SUMO

Things that needed:
1. Install FLOW framework - https://flow.readthedocs.io/en/latest/flow_setup.html
2. Install SUMO -https://sumo.dlr.de/docs/Downloads.php or follow FLOW installation

Then, put traffic_light_grid_test.py into FLOW folder under envs folder (/flow/flow/envs)
Modify _init.py_ of envs folder by adding:
1. from flow.envs.traffic_light_grid_test import TrafficLightGridEnvTest, \
    TrafficLightGridPOEnvTest, TrafficLightGridTestEnvTest, TrafficLightGridBenchmarkEnvTest"
2. in __all__ =
    'TrafficLightGridEnvTest',
    'TrafficLightGridPOEnvTest',
    'TrafficLightGridBenchmarkEnvTest',
    'TrafficLightGridTestEnvTest',
    
Also, replace base.py into FLOW folder under network folder (/flow/flow/networks)
Basically the code inside it is the same with the one FLOW provided, just I added few methods inside to make a functional RL environment for my use case

Lastly, put trafficlightRL folder into FLOW folder (/flow)

Additional Information:
The network used in this project is Jalan Bukit Gambir, Pulau Pinang, Malaysia
