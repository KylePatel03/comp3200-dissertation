import random

"""
Config file for simulation. Accessed by most files.
https://arxiv.org/abs/2002.08423
"""

"""Simulation Parameters"""
NUM_CLIENTS: int = 3
__client_names = ['client_agent' + str(i) for i in range(NUM_CLIENTS)]

CLIENT_FRACTION: float = 0.5
ITERATIONS: int = 2
EPOCHS: int = 1
BATCH_SIZE: int = 10

VERBOSITY: bool = True

"""Latency"""
#SIMULATE_LATENCIES: bool = True
# Define any agent-agent communication latencies here. If none is provided, defaults to zero.

# LATENCY_DICT = {}
# # fill in the rest with zeros:
# if 'server_agent0' not in LATENCY_DICT.keys():
#     LATENCY_DICT['server_agent0'] = {}
#
# for client_name in __client_names:
#     if client_name not in LATENCY_DICT.keys():
#         LATENCY_DICT[client_name] = {client_name2: timedelta(seconds=0.1) for client_name2 in __client_names}
#     LATENCY_DICT[client_name]['server_agent0'] = timedelta(seconds=0.1)
#     LATENCY_DICT['server_agent0'][client_name] = timedelta(seconds=0.1)
#
# LATENCY_DICT['client_agent1'] = {client_name: timedelta(seconds=2.0) for client_name in __client_names}
# LATENCY_DICT['client_agent1']['server_agent0'] = timedelta(seconds=2.0)
# LATENCY_DICT['server_agent0']['client_agent1'] = timedelta(seconds=2.0)
#
# LATENCY_DICT['client_agent0']['server_agent0'] = timedelta(seconds=0.3)
# LATENCY_DICT['server_agent0']['client_agent0'] = timedelta(seconds=0.3)

"""
Additional Constants: likely won't need modification
"""
random.seed(0)
# RANDOM_SEEDS: required for reproducibility of simulation. Seeds every iteration of the training for each client
RANDOM_SEEDS = {client_name: list(random.sample(range(0, 1000000), 100)) for client_name in __client_names}
