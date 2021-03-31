import random

"""
Config file for simulation. Accessed by most files.
https://arxiv.org/abs/2002.08423
"""

"""Simulation Parameters"""
NUM_CLIENTS: int = 4
__client_names = ['client_agent' + str(i) for i in range(NUM_CLIENTS)]

CLIENT_FRACTION: float = 0.5
ITERATIONS: int = 3
EPOCHS: int = 1
BATCH_SIZE: int = 10

VERBOSITY: bool = True

"""Latency"""


"""
Additional Constants: likely won't need modification
"""
random.seed(0)
# RANDOM_SEEDS: required for reproducibility of simulation. Seeds every iteration of the training for each client
RANDOM_SEEDS = {client_name: list(random.sample(range(0, 1000000), 100)) for client_name in __client_names}
