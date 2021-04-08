import datetime
import random

import config
from data_partition import *
from initialiser_edge import InitialiserEdge
from initialiser_vanilla import InitialiserVanilla

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    print('InitialiserVanilla..')
    initialiser = InitialiserVanilla(
        num_clients=config.NUM_CLIENTS,
        client_fraction=config.CLIENT_FRACTION,
        iterations=config.ITERATIONS,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        accuracy_threshold=config.ACCURACY_THRESHOLD,
    )
    a = datetime.datetime.now()
    # initialiser.run_simulation(iterations=config.ITERATIONS)
    b = datetime.datetime.now()
